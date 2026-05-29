import json
import requests
import os
import certifi
import re
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from .config import BASE_URL, USERNAME, PASSWORD, PROJECT_NAME, TP_API_TOKEN
from database import get_cached_comments, save_comments, get_cached_entity_type
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


def _find_local(parent, local_name):
    """Find first descendant element by local tag name (namespace-agnostic)."""
    for el in parent.iter():
        tag = el.tag
        if '}' in tag:
            tag = tag.split('}', 1)[1]
        if tag == local_name:
            return el
    return None


def _get_local_text(parent, local_name):
    """Get text content of first descendant with matching local tag name."""
    el = _find_local(parent, local_name)
    return el.text if el is not None else None


def _get_local_attr(parent, local_name, attr):
    """Get attribute from first descendant element by local tag name."""
    el = _find_local(parent, local_name)
    if el is not None:
        return el.get(attr)
    return None


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

    fetched_at = datetime.now().isoformat()
    return all_comments, fetched_at, True


def get_entity_data(entity_id: int, entity_type: str) -> dict | None:
    v1_base = BASE_URL.replace("/api/v2", "/api/v1")
    v1_path = V2_TO_V1_ENDPOINT.get(entity_type)
    if not v1_path:
        logger.debug("No v1 endpoint mapping for entity type '%s'", entity_type)
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

            try:
                root = ET.fromstring(r.text)
            except ET.ParseError:
                logger.warning("Failed to parse XML for %s %s", entity_type, entity_id)
                return None

            result = {}

            es = _find_local(root, "EntityState")
            if es is not None:
                name = es.get("Name") or _get_local_text(es, "Name")
                if name:
                    result["entity_state"] = name
                es_id = es.get("Id") or _get_local_text(es, "Id")
                if es_id:
                    try:
                        result["entity_state_id"] = int(es_id)
                    except ValueError:
                        pass

            desc = _find_local(root, "Description")
            if desc is not None and desc.text:
                result["description"] = desc.text if desc.text != "true" else ""

            cd = _find_local(root, "CreateDate")
            if cd is not None and cd.text:
                result["create_date"] = cd.text

            proj = _find_local(root, "Project")
            if proj is not None:
                pid = proj.get("Id") or _get_local_text(proj, "Id")
                if pid:
                    try:
                        result["project_id"] = int(pid)
                    except ValueError:
                        pass
                pname = proj.get("Name") or _get_local_text(proj, "Name")
                if pname:
                    result["project_name"] = pname

            cf = _find_local(root, "CustomFields")
            if cf is not None:
                custom_fields = {}
                for field_el in cf:
                    tag = field_el.tag
                    if '}' in tag:
                        tag = tag.split('}', 1)[1]
                    if tag != "Field":
                        continue
                    fname = field_el.get("Name") or _get_local_text(field_el, "Name")
                    fvalue = field_el.get("Value") or _get_local_text(field_el, "Value")
                    if fname and fvalue:
                        val = fvalue.strip()
                        if val and val != "true":
                            custom_fields[fname] = val
                        elif val == "true":
                            custom_fields[fname] = "Yes"
                if custom_fields:
                    result["custom_fields"] = custom_fields
                    col_map = {
                        "client": ("Client", "client"),
                        "product": ("Product", "product"),
                        "release_version": ("Release Version", "release_version"),
                        "site": ("Site", "site"),
                        "customer_ref": ("CustomerRef",),
                        "internal_priority": ("Internal Priority",),
                        "support_level": ("Support Level",),
                        "next_action": ("Next Action",),
                        "paid_work": ("Paid Work",),
                        "downtime": ("Downtime",),
                        "out_of_hours": ("Out of hours",),
                        "customer_chased_date": ("Customer Chased date",),
                        "stop_feedback_request": ("Stop Feedback Request",),
                    }
                    for col, keys in col_map.items():
                        for k in keys:
                            v = custom_fields.get(k)
                            if v:
                                result[col] = v
                                break

            if not result:
                logger.warning("No data extracted from v1 API response for %s %s", entity_type, entity_id)
                return None

            return result
        except requests.RequestException:
            if attempt == 2:
                logger.warning("All attempts failed fetching entity data for %s %s", entity_type, entity_id)
                return None
        except Exception:
            logger.warning("Unexpected error in get_entity_data for %s %s", entity_type, entity_id)
            return None

    return None


def get_relations(entity_id: int) -> list[dict]:
    relations = []
    directions = [("Master.Id", entity_id), ("Slave.Id", entity_id)]
    seen_pairs = set()

    for where_field, eid in directions:
        url = f"{BASE_URL}/Relation"
        params = {"where": f"{where_field} = {eid}", "take": 100}
        while url:
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
            params = {}

    return relations


def refresh_entity_metadata(entity_id: int, depth: int = 0, seen: set | None = None) -> None:
    """Fetch and persist entity metadata (type, custom fields, relations) for an entity.

    Operates one level deep by default — also saves metadata for direct relations.
    Uses a seen set to avoid cycles.
    Checks the DB cache first and skips API calls if all data already exists.
    """
    if seen is None:
        seen = set()
    if entity_id in seen or entity_id is None:
        return
    seen.add(entity_id)

    entity_type = get_cached_entity_type(entity_id)
    if not entity_type:
        entity_type = get_entity_type(entity_id)

    if entity_type:
        if entity_type not in V2_TO_V1_ENDPOINT:
            logger.debug("Skipping metadata for unsupported type '%s' %s", entity_type, entity_id)
        else:
            from database import get_entity_data as _get_db_ed, save_entity_data as _save_ed
            from database import get_relations as _get_db_rel, save_relations as _save_rel

            if not _get_db_ed(entity_id):
                entity_data = get_entity_data(entity_id, entity_type)
                if entity_data:
                    try:
                        _save_ed(entity_id, entity_type=entity_type, **entity_data)
                    except Exception:
                        logger.warning("Failed to save entity_data for %s", entity_id)

            relations = _get_db_rel(entity_id)
            if not relations:
                try:
                    relations = get_relations(entity_id)
                    if relations:
                        _save_rel(entity_id, entity_type, relations)
                except Exception:
                    logger.warning("Failed to save relations for %s", entity_id)

            if depth == 0 and relations:
                for rel in relations:
                    other_id = rel.get("related_entity_id")
                    if other_id and other_id not in seen:
                        refresh_entity_metadata(other_id, depth=1, seen=seen)
