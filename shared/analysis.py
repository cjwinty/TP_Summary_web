import json
import re

from . import config
from database import get_prompt as db_get_prompt, DEFAULT_PROMPTS


def get_llm():
    return config.initialise_llm()


def clean_html(text):
    if not text:
        return ""

    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)

    text = re.sub(r'</p\s*>', '\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</div\s*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<hr\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</tr\s*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</td\s*>', '  ', text, flags=re.IGNORECASE)
    text = re.sub(r'</th\s*>', '  ', text, flags=re.IGNORECASE)

    text = re.sub(r'<li[^>]*>', '\n• ', text, flags=re.IGNORECASE)
    text = re.sub(r'</li\s*>', '', text, flags=re.IGNORECASE)

    text = re.sub(r'</h[1-6]\s*>', '\n\n', text, flags=re.IGNORECASE)

    text = re.sub(r'<[^>]+>', '', text)

    text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
    text = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), text)
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&apos;', "'")
    text = text.replace('&hellip;', '...')
    text = text.replace('&mdash;', '—')
    text = text.replace('&ndash;', '–')
    text = text.replace('&lsquo;', '\u2018')
    text = text.replace('&rsquo;', '\u2019')
    text = text.replace('&ldquo;', '\u201c')
    text = text.replace('&rdquo;', '\u201d')

    lines = [line.rstrip() for line in text.splitlines()]
    result = []
    prev_blank = False
    for line in lines:
        is_blank = line.strip() == ''
        if is_blank and prev_blank:
            continue
        result.append(line)
        prev_blank = is_blank

    return '\n'.join(result).strip()


def clean_html_tags(text):
    return clean_html(text)


def deduplicate_comment_dicts(comments_list):
    seen = set()
    unique_lines = []
    for comment in comments_list:
        text = comment.get("text", "")
        for line in text.split('\n'):
            line = line.strip()
            if line and line not in seen and len(line) > 5:
                seen.add(line)
                unique_lines.append(line)
    return unique_lines


def deduplicate_text(text):
    seen = set()
    result = []
    for line in text.split('\n'):
        line = line.strip()
        if line and line not in seen and len(line) > 3:
            seen.add(line)
            result.append(line)
    return '\n'.join(result)


def _safe_format(template, **kwargs):
    class DefaultDict(dict):
        def __missing__(self, key):
            return ""
    return template.format_map(DefaultDict(kwargs))


def extract_issues_batch(comments_list, entity_type="Request", entity_id=None):
    if not comments_list:
        return []

    base_prompt = get_prompt("extract_issues")
    all_issues = []
    llm = get_llm()

    for comment_text in comments_list:
        prompt = _safe_format(base_prompt, entity_type=entity_type,
                              entity_id=str(entity_id or ""),
                              comments=comment_text[:3000])

        for retry in range(3):
            try:
                content = llm.generate(prompt, temperature=0.2)
                if content and content.lower() != "unclassified":
                    categories = [c.strip() for c in content.split(",")]
                    all_issues.extend(categories)
                break
            except Exception as e:
                if retry == 2:
                    print(f"Error extracting issues: {e}")
                else:
                    print(f"Retry {retry + 1}/3...")

    return all_issues


def get_prompt(name):
    try:
        result = db_get_prompt(name)
        if result:
            return result["content"]
    except Exception:
        pass
    return DEFAULT_PROMPTS.get(name, "")


def summarise_comments(comments_text, entity_id=None, entity_type="Request", project_name=""):
    base_prompt = get_prompt("summarise")
    prompt = _safe_format(
        base_prompt,
        entity_type=entity_type,
        entity_id=str(entity_id or ""),
        project_name=project_name,
        comments_text=comments_text,
    )
    llm = get_llm()
    try:
        return llm.generate(prompt, temperature=0.3)
    except Exception as e:
        return f"Error: {e}"


def summarise_batch(texts, batch_size=1, entity_type="Request", entity_ids=None):
    results = []
    base_prompt = get_prompt("summarise")
    llm = get_llm()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        eid = str(entity_ids[i]) if entity_ids and i < len(entity_ids) else ""

        prompt = _safe_format(
            base_prompt,
            entity_type=entity_type,
            entity_id=eid,
            project_name="",
            comments_text="",
        )
        for item in batch:
            prompt += "\n" + item + "\n"

        for retry in range(3):
            try:
                content = llm.generate(prompt, temperature=0.3)
                results.append(content)
                break
            except Exception as e:
                if retry == 2:
                    results.append(f"Error processing: {e}")
                else:
                    print(f"Retry {retry + 1}/3...")

        if i + batch_size < len(texts):
            print(f"Processed {i + batch_size}/{len(texts)} requests...")

    return results


def refine_search_query(query):
    base_prompt = get_prompt("refine_search")
    prompt = _safe_format(base_prompt, query=query)
    try:
        llm = get_llm()
        content = llm.generate(prompt, temperature=0.3)
        terms = [term.strip() for term in content.split('\n') if term.strip()]
        return [query] + terms
    except Exception as e:
        return [query]


def summarise_search_results(matches, query, custom_prompt=""):
    if not matches:
        return "No matching results found."

    results_text = "\n\n".join([
        f"[#{m['request_id']} ({m.get('entity_type', 'Entity')} / {m['source']})]\n{m['text'][:500]}"
        for m in matches[:20]
    ])

    base_prompt = get_prompt("summarise_search")
    default_prompt = _safe_format(
        base_prompt,
        query=query,
        match_count=len(matches),
        results_text=results_text,
    )

    if custom_prompt:
        prompt = default_prompt + "\n\n" + custom_prompt
    else:
        prompt = default_prompt

    llm = get_llm()
    try:
        return llm.generate(prompt, temperature=0.3)
    except Exception as e:
        return f"Error generating summary: {e}"
