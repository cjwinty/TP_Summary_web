"""
prompt_chain_executor.py
========================
Core execution engine for Prompt Chains — web-adapted copy.
"""
import re
import time
import datetime
from typing import Callable, Any

from . import config
from database import (
    get_cached_comments, get_cached_entity_type, get_entity_data,
    get_summary, get_custom_fields, search_cached_issues_by_product_keyword,
)


def _format_date(date_val):
    if not date_val:
        return "Unknown"
    if isinstance(date_val, str) and date_val.startswith("/Date("):
        match = re.search(r"/Date\((\d+)", date_val)
        if match:
            try:
                ts = int(match.group(1)) / 1000
                return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            except (ValueError, OSError):
                pass
    return str(date_val)


def _extract_keywords_from_search_terms(search_terms_output: str) -> list[str]:
    if not search_terms_output:
        return []

    keywords = []

    primary_match = re.search(r"PRIMARY SEARCH TERM:\s*(.+?)(?:\n|$)", search_terms_output, re.IGNORECASE)
    if primary_match:
        primary = primary_match.group(1).strip()
        if primary and primary.lower() != "none":
            keywords.append(primary)

    secondary_match = re.search(r"SECONDARY SEARCH TERMS:(.+?)(?:COMPONENT|$)", search_terms_output, re.IGNORECASE | re.DOTALL)
    if secondary_match:
        secondary_text = secondary_match.group(1)
        for line in secondary_text.split("\n"):
            line = line.strip().lstrip("-•* ")
            if line and line.lower() != "none" and len(line) > 1:
                keywords.append(line)

    if not keywords:
        words = re.findall(r'\b[A-Z][a-z]+\b', search_terms_output)
        keywords = [w for w in words if w.lower() not in ('none', 'primary', 'secondary', 'search', 'term', 'component', 'filter', 'strict', 'matching', 'rule')]

    return keywords


def _auto_search_similar_issues(context: dict, search_terms_output: str) -> str:
    keywords = _extract_keywords_from_search_terms(search_terms_output)
    if not keywords:
        return "No search terms generated."

    product = context.get("cached_product", "")
    if product and product.lower() == "not recorded":
        product = None

    try:
        results = search_cached_issues_by_product_keyword(product, keywords, limit=10)
        if not results and product:
            results = search_cached_issues_by_product_keyword(None, keywords, limit=10)
    except Exception as e:
        return f"Error searching database: {e}"

    if not results:
        if product:
            return f"No similar issues found for product '{product}' with keywords: {', '.join(keywords)}"
        else:
            return f"No similar issues found with keywords: {', '.join(keywords)}"

    formatted = []
    for i, result in enumerate(results, 1):
        et = get_cached_entity_type(result['request_id']) or "Request"
        formatted.append(f"SIMILAR ISSUE #{i} ({et} ID: {result['request_id']})")
        formatted.append(f"Match Reason: {result['match_reason']}")
        formatted.append(f"Product: {result.get('product', 'Unknown')}")
        formatted.append("---")
        formatted.append(result["text"])
        formatted.append("\n")

    return "\n".join(formatted)


_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def render_template(template: str, context: dict) -> str:
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        return str(context.get(key, match.group(0)))

    return _PLACEHOLDER_RE.sub(_replace, template)


def _call_llm(prompt: str, model: str | None = None,
               temperature: float = 0.3, timeout: int = 300) -> str:
    llm = config.initialize_llm()
    try:
        content = llm.generate(prompt, temperature=temperature)
        if not content:
            raise RuntimeError("LLM returned an empty response")
        return content
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc


def _handle_db_query(step: dict, context: dict) -> str:
    query_type = step.get("prompt_template", "").strip().lower()

    if query_type == "search_keywords":
        search_terms = context.get("search_terms", "")
        if not search_terms:
            return "No search terms available in context."

        keywords = _extract_keywords_from_search_terms(search_terms)
        if not keywords:
            return "No keywords could be extracted from search terms."

        product = context.get("cached_product", "")
        if product and product.lower() == "not recorded":
            product = None

        try:
            results = search_cached_issues_by_product_keyword(product, keywords, limit=10)
            if not results and product:
                results = search_cached_issues_by_product_keyword(None, keywords, limit=10)
        except Exception as e:
            return f"Error searching database: {e}"

        if not results:
            if product:
                return f"No similar issues found for product '{product}' with keywords: {', '.join(keywords)}"
            return f"No similar issues found with keywords: {', '.join(keywords)}"

        formatted = []
        for i, result in enumerate(results, 1):
            et = get_cached_entity_type(result['request_id']) or "Request"
            formatted.append(f"SIMILAR ISSUE #{i} ({et} ID: {result['request_id']})")
            formatted.append(f"Match Reason: {result['match_reason']}")
            formatted.append(f"Product: {result.get('product', 'Unknown')}")
            formatted.append("---")
            formatted.append(result["text"])
            formatted.append("\n")

        return "\n".join(formatted)

    request_id = context.get("input", "").strip()

    if not request_id:
        raise RuntimeError("No request ID provided in input")

    try:
        request_id = int(request_id)
    except ValueError:
        raise RuntimeError(f"Invalid request ID: {request_id}")

    result_parts = []

    if query_type in ("get_comments", "get_all"):
        comments, _ = get_cached_comments(request_id)
        if comments:
            result_parts.append("COMMENTS:\n---")
            for i, c in enumerate(comments[:50]):
                date = _format_date(c.get("date"))
                text = c.get("text", "")
                result_parts.append(f"[{date}] COMMENT {i+1}:\n{text}\n---")
        else:
            result_parts.append("COMMENTS: None found")

    if query_type in ("get_summary", "get_all"):
        summary, created_at = get_summary(request_id)
        if summary:
            result_parts.append(f"\nEXISTING SUMMARY (created {created_at}):\n{summary}")
        else:
            result_parts.append("\nEXISTING SUMMARY: None available")

    if query_type in ("get_custom_fields", "get_all"):
        fields, _ = get_custom_fields(request_id)
        if fields:
            result_parts.append("\nCUSTOM FIELDS:")
            for fname, fvalue in fields.items():
                result_parts.append(f"  {fname}: {fvalue}")
        else:
            result_parts.append("\nCUSTOM FIELDS: None recorded")

    if not result_parts:
        raise RuntimeError(f"Unknown query type: {query_type}")

    return "\n".join(result_parts)


def execute_chain(
    chain_id: int,
    initial_input: str,
    model: str | None = None,
    temperature: float = 0.3,
    on_step_complete: Callable[[int, str, dict], None] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    db_get_chain=None,
    db_create_run=None,
    db_update_run_step=None,
    db_finish_run=None,
) -> dict:
    if db_get_chain is None:
        from database import get_chain as db_get_chain
    if db_create_run is None:
        from database import create_run as db_create_run
    if db_update_run_step is None:
        from database import update_run_step as db_update_run_step
    if db_finish_run is None:
        from database import finish_run as db_finish_run

    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    chain = db_get_chain(chain_id)
    if not chain:
        return {
            "run_id": None, "chain_id": chain_id,
            "status": "failed", "final_output": None,
            "context": {}, "steps": [],
            "error": f"Chain {chain_id} not found",
        }

    steps = chain.get("steps", [])
    if not steps:
        return {
            "run_id": None, "chain_id": chain_id,
            "status": "failed", "final_output": None,
            "context": {"input": initial_input}, "steps": [],
            "error": "Chain has no steps",
        }

    run_id = db_create_run(chain_id, initial_input)

    context: dict[str, Any] = {"input": initial_input}

    cached_data_warning = None
    try:
        request_id = int(initial_input.strip())
        comments, _ = get_cached_comments(request_id)
        fields, _ = get_custom_fields(request_id)
        entity_type = get_cached_entity_type(request_id) or "Request"

        if not comments and not fields:
            cached_data_warning = f"No cached data found for request {request_id}. Please download it first using the Search Cache."

        context["cached_entity_type"] = entity_type
        context["cached_entity_id"] = str(request_id)

        if comments:
            context["cached_comments"] = "\n\n".join([
                f"[{_format_date(c.get('date'))}] {c.get('text', '')}"
                for c in comments[:50]
            ])
        else:
            context["cached_comments"] = "No cached comments found"

        summary, _ = get_summary(request_id)
        if summary:
            context["cached_summary"] = summary
        else:
            context["cached_summary"] = "No existing summary"

        if fields:
            context["cached_client"] = fields.get("Client", "Not recorded")
            context["cached_product"] = fields.get("Product", "Not recorded")
            context["cached_release_version"] = fields.get("Release Version", "Not recorded")
            context["cached_site"] = fields.get("Site", "Not recorded")
        else:
            context["cached_client"] = "Not recorded"
            context["cached_product"] = "Not recorded"
            context["cached_release_version"] = "Not recorded"
            context["cached_site"] = "Not recorded"

        entity_data = get_entity_data(request_id)
        if entity_data:
            context["cached_description"] = entity_data.get("description") or ""
            context["cached_create_date"] = entity_data.get("create_date") or ""
            context["cached_entity_state"] = entity_data.get("entity_state") or ""
            custom_fields = entity_data.get("custom_fields") or {}
            context["cached_custom_fields"] = "\n".join(
                f"{k}: {v}" for k, v in custom_fields.items()
            ) if custom_fields else "No custom fields"
        else:
            context["cached_description"] = ""
            context["cached_create_date"] = ""
            context["cached_entity_state"] = ""
            context["cached_custom_fields"] = "No cached data"

    except ValueError:
        context["cached_entity_type"] = "Unknown"
        context["cached_entity_id"] = initial_input
        context["cached_comments"] = "No input (not a request ID)"
        context["cached_summary"] = "No input (not a request ID)"
        context["cached_client"] = "No input (not a request ID)"
        context["cached_product"] = "No input (not a request ID)"
        context["cached_release_version"] = "No input (not a request ID)"
        context["cached_site"] = "No input (not a request ID)"
        context["cached_description"] = "No input (not a request ID)"
        context["cached_create_date"] = "No input (not a request ID)"
        context["cached_entity_state"] = "No input (not a request ID)"
        context["cached_custom_fields"] = "No input (not a request ID)"

    if cached_data_warning:
        _progress(cached_data_warning)

    step_results: list[dict] = []
    last_output: str = initial_input
    chain_error: str | None = None

    for step in steps:
        order   = step["step_order"]
        name    = step.get("name") or f"Step {order}"
        out_var = step["output_variable"]

        _progress(f"Running step {order}/{len(steps)}: {name}…")

        step_result: dict = {
            "step_order":  order,
            "name":        name,
            "input_sent":  None,
            "output":      None,
            "status":      "pending",
            "duration_ms": None,
            "error":       None,
        }

        t_start = time.monotonic()

        step_type = step.get("step_type", "llm")
        if step_type == "db_query":
            try:
                output_text = _handle_db_query(step, context)
                duration_ms = int((time.monotonic() - t_start) * 1000)
                step_result["input_sent"] = f"DB Query: {step.get('prompt_template', '')}"
            except Exception as e:
                duration_ms = int((time.monotonic() - t_start) * 1000)
                error_msg = str(e)
                step_result.update({
                    "status":      "failed",
                    "error":       error_msg,
                    "duration_ms": duration_ms,
                })
                chain_error = f"Step {order} ({name}) failed: {error_msg}"
                db_update_run_step(
                    run_id=run_id,
                    step_id=step["id"],
                    step_order=order,
                    input_sent=step_result["input_sent"],
                    output_received=None,
                    status="failed",
                    error=error_msg,
                    duration_ms=duration_ms,
                )
                step_results.append(step_result)
                _progress(f"Chain failed at step {order}: {error_msg}")
                break
        else:
            merged_context = {**step.get("variables", {}), **context}

            rendered_prompt = render_template(step["prompt_template"], merged_context)
            step_result["input_sent"] = rendered_prompt

            try:
                output_text = _call_llm(
                    rendered_prompt,
                    model=model,
                    temperature=temperature,
                )
                duration_ms = int((time.monotonic() - t_start) * 1000)

                context[out_var] = output_text
                last_output = output_text

                step_result.update({
                    "output":      output_text,
                    "status":      "completed",
                    "duration_ms": duration_ms,
                })

                db_update_run_step(
                    run_id=run_id,
                    step_id=step["id"],
                    step_order=order,
                    input_sent=rendered_prompt,
                    output_received=output_text,
                    status="completed",
                    duration_ms=duration_ms,
                )

                if out_var == "search_terms":
                    similar_issues = _auto_search_similar_issues(context, output_text)
                    context["similar_issues"] = similar_issues

                if on_step_complete:
                    on_step_complete(order, output_text, dict(context))

            except RuntimeError as exc:
                duration_ms = int((time.monotonic() - t_start) * 1000)
                error_msg = str(exc)
                step_result.update({
                    "status":      "failed",
                    "error":       error_msg,
                    "duration_ms": duration_ms,
                })
                chain_error = f"Step {order} ({name}) failed: {error_msg}"

                db_update_run_step(
                    run_id=run_id,
                    step_id=step["id"],
                    step_order=order,
                    input_sent=rendered_prompt,
                    output_received=None,
                    status="failed",
                    error=error_msg,
                    duration_ms=duration_ms,
                )

                step_results.append(step_result)
                _progress(f"Chain failed at step {order}: {error_msg}")
                break

        step_results.append(step_result)

    final_status = "failed" if chain_error else "completed"
    final_output = None if chain_error else last_output

    db_finish_run(
        run_id=run_id,
        status=final_status,
        final_output=final_output,
        error=chain_error,
    )

    _progress("Chain complete." if not chain_error else f"Chain failed: {chain_error}")

    return {
        "run_id":       run_id,
        "chain_id":     chain_id,
        "status":       final_status,
        "final_output": final_output,
        "context":      context,
        "steps":        step_results,
        "error":        chain_error,
    }


def execute_chain_by_name(
    chain_name: str,
    initial_input: str,
    **kwargs,
) -> dict:
    from database import list_chains
    chains = list_chains()
    match = next((c for c in chains if c["name"] == chain_name), None)
    if not match:
        raise ValueError(f"No chain named '{chain_name}'")
    return execute_chain(match["id"], initial_input, **kwargs)
