import os
import jinja2
from starlette.templating import _TemplateResponse
from shared.config import VERSION

_dir = os.path.join(os.path.dirname(__file__), "templates")
_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(_dir),
    autoescape=jinja2.select_autoescape(),
    auto_reload=False,
    cache_size=-1,
)
_env.globals["version"] = VERSION


class Templates:
    def get_template(self, name: str) -> jinja2.Template:
        return _env.get_template(name)

    def TemplateResponse(self, name, context, status_code=200, headers=None, media_type=None):
        template = self.get_template(name)
        return _TemplateResponse(template, context, status_code=status_code, headers=headers, media_type=media_type)


templates = Templates()
