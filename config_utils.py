from pathlib import Path


def _parse_scalar(value):
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def load_config(path):
    """
    Tiny YAML loader for the simple nested config files in configs/*.yaml.
    Uses PyYAML if available; otherwise supports key/value indentation.
    """
    path = Path(path)
    text = path.read_text()

    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:
        pass

    root = {}
    stack = [(0, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent < stack[-1][0]:
            stack.pop()

        current = stack[-1][1]

        if value == "":
            child = {}
            current[key] = child
            stack.append((indent + 2, child))
        else:
            current[key] = _parse_scalar(value)

    return root


def cfg_get(cfg, path, default=None):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur