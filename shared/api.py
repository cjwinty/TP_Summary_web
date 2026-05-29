import json
import requests
import os
import certifi
import re
import logging
from datetime import datetime
from .config import BASE_URL, USERNAME, PASSWORD, PROJECT_NAME, TP_API_TOKEN
from database import get_cached_comments, save_comments
from .analysis import clean_html

logger = logging.getLogger(__name__)

CA_BUNDLE = certifi.where()

V2_TO_V1_ENDPOINT = {
    "Request": "requests",
    "UserStory": "userstories",
    "Bug": "bugs",
    "Feature": "features",
    "Epic": "epics",
    "Task": "tasks",
    "Impediment": "impediments",
    "PortfolioEpic": "portfolioepics",
}


def _get_auth():
    if TP_API_TOKEN:
        logger.info("Using API token authentication")
        return None, None, {"access_token": TP_API_TOKEN}
    if USERNAME and PASSWORD:
        logger.info("Using basic auth (username/password)")
        return (USERNAME, PASSWORD), None, {}
    logger.warning("No API credentials configured — set TP_API_TOKEN or TP_USERNAME/TP_PASSWORD in .env")
    return None, None, {}


def make_request(url, params, retries=3):
    for attempt in range(retries):
        try:
            auth, headers, extra_params = _get_auth()
            merged_params = {**params, **extra_params} if extra_params else params
            r = requests.get(
                url, params=merged_params, auth=auth, headers=headers or None,
                timeout=30, verify=CA_BUNDLE,
            )
            try:
                data = r.json()
            except ValueError:
                snippet = r.text[:200] if r.text else ""
                logger.warning("API returned non-JSON response (HTTP %d): %s", r.status_code, snippet)
                if attempt == retries - 1:
                    return None
                continue
            if "Status" in data and data["Status"] == "BadRequest":
                logger.error("API Error: %s", data.get("Message", "Unknown"))
                return None
            return data
        except requests.RequestException as e:
            logger.warning("Request failed (attempt %d/%d): %s", attempt + 1, retries, e)
            if attempt == retries - 1:
                return None
    return None


def get_entity_type(entity_id: int) -> str | None:
    url = f"{BASE_URL}/Assignables/{entity_id}"
    data = make_request(url, {})
    if data and data.get("items"):
        return data["items"][0].get("resourceType")
    return None


def get_entities_by_project(project_id, entity_type, take=200):
    all_items = []
    url = f"{BASE_URL}/{entity_type}"
    params = {"where": f"Project.Id={project_id}", "take": take}
    while url:
        data = make_request(url, params)
        if not data:
            break
        items = data.get("items", [])
        all_items.extend(items)
        url = data.get("next")
        params = {}
    return all_items


def get_all_projects():
    data = make_request(f"{BASE_URL}/Project", {"take": 200})
    if data:
        return data.get("items", [])
    return []


def get_comments(entity_id, use_cache=True):
    if use_cache:
        cached, fetched_at = get_cached_comments(entity_id)
        if cached is not None:
            return cached, fetched_at, False

    all_comments = []
    api_error = False

    comments_url = f"{BASE_URL}/Comments"
    dates_url = f"{BASE_URL}/Comments"
    comments_params = {"where": f"General.Id = {entity_id}", "select": "Description", "take": 100}
    dates_params = {"where": f"General.Id = {entity_id}", "select": "CreateDate", "take": 100}

    comments_by_idx = {}
    dates_by_idx = {}
    max_idx = 0

    while comments_url:
        data = make_request(comments_url, comments_params)
        if not data:
            api_error = True
            break
        items = data.get("items", [])
        for i, item in enumerate(items):
            if isinstance(item, str) and item.strip():
                comments_by_idx[max_idx + i] = item
        max_idx += len(items)
        comments_url = data.get("next")
        comments_params = {}

    max_idx = 0
    while dates_url:
        data = make_request(dates_url, dates_params)
        if not data:
            break
        items = data.get("items", [])
        for i, item in enumerate(items):
            if isinstance(item, str) and item.strip():
                dates_by_idx[max_idx + i] = item
        max_idx += len(items)
        dates_url = data.get("next")
        dates_params = {}

    for idx in sorted(comments_by_idx.keys()):
        raw_text = comments_by_idx[idx]
        cleaned_text = clean_html(raw_text)
        if cleaned_text.strip():
            all_comments.append({"text": cleaned_text, "date": dates_by_idx.get(idx)})

    all_comments.sort(key=lambda x: x.get("date") or "")

    if api_error and not all_comments:
        return None, None, True

    if all_comments:
        entity_type = get_entity_type(entity_id)
        save_comments(entity_id, comments=all_comments, entity_type=entity_type)

        try:
            entity_data = get_entity_data(entity_id, entity_type) if entity_type else None
            if entity_data:
                from database import save_entity_data as _save_entity_data
                _save_entity_data(entity_id, entity_type=entity_type, **entity_data)

            relations = get_relations(entity_id)
            if relations:
                from database import save_relations as _save_relations
                _save_relations(entity_id, entity_type or "Request", relations)
        except Exception:
            logger.warning("Failed to save entity metadata for %s", entity_id)

    fetched_at = datetime.now().isoformat()
    return all_comments, fetched_at, True


def get_entity_data(entity_id: int, entity_type: str) -> dict | None:
    v1_base = BASE_URL.replace("/api/v2", "/api/v1")
    v1_path = V2_TO_V1_ENDPOINT.get(entity_type)
    if not v1_path:
        return None
    url = f"{v1_base}/{v1_path}/{entity_id}"
    params = {"include": "[id,customFields]"}

    auth, headers, extra_params = _get_auth()
    merged_params = {**params, **extra_params} if extra_params else params
    for attempt in range(3):
        try:
            r = requests.get(
                url, params=merged_params, auth=auth, headers=headers or None,
                timeout=30, verify=CA_BUNDLE,
            )
            if r.status_code != 200:
                logger.warning("Entity data API returned status %d for %s %s", r.status_code, entity_type, entity_id)
                return None

            xml_text = r.text

            result = {}

            es_match = re.search(r'<EntityState[^>]+Name="([^"]+)"', xml_text)
            if es_match:
                result["entity_state"] = es_match.group(1)
            es_id_match = re.search(r'<EntityState[^>]+Id="(\d+)"', xml_text)
            if es_id_match:
                result["entity_state_id"] = int(es_id_match.group(1))

            desc_match = re.search(r'<Description[^>]*>(.*?)</Description>', xml_text, re.DOTALL)
            if desc_match:
                val = desc_match.group(1)
                result["description"] = val if val != "true" else ""

            cd_match = re.search(r'<CreateDate>([^<]*)</CreateDate>', xml_text)
            if cd_match:
                result["create_date"] = cd_match.group(1)

            proj_match = re.search(r'<Project[^>]+Id="(\d+)"[^>]*Name="([^"]*)"', xml_text)
            if proj_match:
                result["project_id"] = int(proj_match.group(1))
                result["project_name"] = proj_match.group(2)

            start = xml_text.find("<CustomFields>")
            if start >= 0:
                cf_section = xml_text[start:]
                field_pattern = r'<Field\s+Type="([^"]*)">\s*<Name>([^<]*)</Name>\s*<Value>([^<]*)</Value>\s*</Field>'
                matches = re.findall(field_pattern, cf_section, re.DOTALL)
                custom_fields = {}
                for field_type, field_name, field_value in matches:
                    val = field_value.strip()
                    if val and val != "true":
                        custom_fields[field_name] = val
                    elif val == "true":
                        custom_fields[field_name] = "Yes"
                    elif val:
                        pass
                if custom_fields:
                    result["custom_fields"] = custom_fields

            return result if result else None
        except requests.RequestException:
            if attempt == 2:
                return None
        except Exception:
            return None

    return None


def get_relations(entity_id: int) -> list[dict]:
    relations = []
    directions = [("Master.Id", entity_id), ("Slave.Id", entity_id)]
    seen_pairs = set()

    for where_field, eid in directions:
        url = f"{BASE_URL}/Relation"
        while url:
            params = {"where": f"{where_field} = {eid}", "take": 100}
            data = make_request(url, params)
            if not data:
                break
            for item in data.get("items", []):
                rel_id = item.get("id")
                if not rel_id:
                    continue
                r_m = make_request(f"{BASE_URL}/Relation/{rel_id}", {"select": "master"})
                r_s = make_request(f"{BASE_URL}/Relation/{rel_id}", {"select": "slave"})
                master = (r_m.get("items") or [None])[0] if r_m else None
                slave = (r_s.get("items") or [None])[0] if r_s else None
                if not master or not slave:
                    continue

                if master.get("id") == entity_id:
                    other = slave
                else:
                    other = master

                other_id = other.get("id")
                pair_key = (rel_id, other_id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                other_type = None
                other_state = None
                if other_id:
                    ot = get_entity_type(other_id)
                    other_type = ot
                    if ot:
                        v1_path = V2_TO_V1_ENDPOINT.get(ot)
                        if v1_path:
                            v1_base = BASE_URL.replace("/api/v2", "/api/v1")
                            try:
                                auth, hdrs, xtra = _get_auth()
                                r_v1 = requests.get(
                                    f"{v1_base}/{v1_path}/{other_id}",
                                    params=xtra, auth=auth, headers=hdrs,
                                    timeout=30, verify=CA_BUNDLE,
                                )
                                if r_v1.status_code == 200:
                                    es = re.search(r'<EntityState[^>]+Name="([^"]+)"', r_v1.text)
                                    if es:
                                        other_state = es.group(1)
                            except Exception:
                                pass

                relations.append({
                    "related_entity_id": other_id,
                    "related_entity_type": other_type,
                    "related_entity_name": other.get("name"),
                    "related_entity_state": other_state,
                    "relation_id": rel_id,
                })

            url = data.get("next")

    return relations
