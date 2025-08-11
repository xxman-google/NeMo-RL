"""Utility functions for HTML rendering.

The file is adapted from: https://github.com/openai/simple-evals.
"""

from typing import Any

import jinja2

Message = dict[str, Any]  # keys role, content

HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ preformatted_message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ preformatted_message_to_html(next_message) | safe }}
<h3>Results</h3>
<pre>Correct Answer: {{ correct_answer }} </pre>
<pre>Extracted Answer: {{ extracted_answer }} </pre>
<p>Score: {{ score }}</p>
"""

CODE_HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ preformatted_message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ message_to_html(next_message) | safe }}
<h3>Results</h3>
<p>Correct Answer: <github-md> {{ correct_answer }} </github-mb></p>
<p>Extracted Answer: <github-md> {{ extracted_answer }} </github-mb></p>
<p>Score: {{ score }}</p>
"""

MATH_HTML_JINJA = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ preformatted_message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ message_to_html(next_message) | safe }}
<h3>Results</h3>
<p>Correct Answer: {{ correct_answer }} </p>
<p>Extracted Answer: {{ extracted_answer }} </p>
<p>Score: {{ score }}</p>
"""

_message_template = """
<div class="message {{ role }}">
    <div class="role">
    {{ role }}
    {% if variant %}<span class="variant">({{ variant }})</span>{% endif %}
    </div>
    <div class="content">
    <p><github-md> {{ content }} </github-md></p>
    </div>
</div>
"""

_preformatted_message_template = """
<div class="message {{ role }}">
    <div class="role">
    {{ role }}
    {% if variant %}<span class="variant">({{ variant }})</span>{% endif %}
    </div>
    <div class="content">
    <pre>{{ content }}</pre>
    </div>
</div>
"""

_markdown_message_template = """
<div class="message {{ role }}">
    <div class="role">
    {{ role }}
    {% if variant %}<span class="variant">({{ variant }})</span>{% endif %}
    </div>
    <div class="content">
    <p><github-md>{{ content }}</github-md></p>
    </div>
</div>
"""

_report_template = """<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
        <style>
            .message {
                padding: 8px 16px;
                margin-bottom: 8px;
                border-radius: 4px;
            }
            .message.user {
                background-color: #B2DFDB;
                color: #00695C;
            }
            .message.assistant {
                background-color: #B39DDB;
                color: #4527A0;
            }
            .message.system {
                background-color: #EEEEEE;
                color: #212121;
            }
            .role {
                font-weight: bold;
                margin-bottom: 4px;
            }
            .variant {
                color: #795548;
            }
            table, th, td {
                border: 1px solid black;
            }
            pre {
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
    {% if metrics %}
    <h1>Metrics</h1>
    <table>
    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>
    <tr>
        <td><b>Score</b></td>
        <td>{{ score | float | round(3) }}</td>
    </tr>
    {% for name, value in metrics.items() %}
    <tr>
        <td>{{ name }}</td>
        <td>{{ value }}</td>
    </tr>
    {% endfor %}
    </table>
    {% endif %}
    <h1>Examples</h1>
    {% for html in htmls %}
    {{ html | safe }}
    <hr>
    {% endfor %}
    </body>
    <script src="https://cdn.jsdelivr.net/gh/MarketingPipeline/Markdown-Tag/markdown-tag-GitHub.js"></script>
</html>
"""

jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    undefined=jinja2.StrictUndefined,
    autoescape=jinja2.select_autoescape(["html", "xml"]),
)


def make_report_from_example_htmls(htmls: list[str]) -> str:
    """Create a standalone HTML report from a list of example htmls."""
    return jinja_env.from_string(_report_template).render(
        score=None, metrics={}, htmls=htmls
    )


def message_to_html(message: Message) -> str:
    """Generate HTML snippet (inside a <div>) for a message."""
    return jinja_env.from_string(_message_template).render(
        role=message["role"],
        content=message["content"],
        variant=message.get("variant", None),
    )


def markdown_message_to_html(message: Message) -> str:
    """Generate HTML snippet (inside a <div>) for a message."""
    return jinja_env.from_string(_markdown_message_template).render(
        role=message["role"],
        content=message["content"],
        variant=message.get("variant", None),
    )


def preformatted_message_to_html(message: Message) -> str:
    """Generate HTML snippet (inside a <div>) for a message."""
    return jinja_env.from_string(_preformatted_message_template).render(
        role=message["role"],
        content=message["content"],
        variant=message.get("variant", None),
    )


jinja_env.globals["message_to_html"] = message_to_html
jinja_env.globals["markdown_message_to_html"] = markdown_message_to_html
jinja_env.globals["preformatted_message_to_html"] = preformatted_message_to_html


class BaseRenderTemplate:
    def __init__(self):
        self._template = jinja_env.from_string(HTML_JINJA)

    def render(
        self,
        prompt: str,
        response: str,
        score: float,
        correct_answer: str,
        extracted_answer: str,
    ):
        prompt_messages = [{"content": prompt, "role": "user"}]
        next_message = dict(content=response, role="assistant")
        return self._template.render(
            prompt_messages=prompt_messages,
            next_message=next_message,
            score=score,
            correct_answer=correct_answer,
            extracted_answer=extracted_answer,
        )


class MathRenderTemplate(BaseRenderTemplate):
    def __init__(self):
        self._template = jinja_env.from_string(MATH_HTML_JINJA)


class CodeRenderTemplate:
    def __init__(self, lang: str = "python"):
        self._template = jinja_env.from_string(CODE_HTML_JINJA)
        self._lang = lang

    def render(
        self,
        prompt: str,
        response: str,
        score: float,
        correct_answer: str,
        extracted_answer: str,
    ):
        prompt_messages = [{"content": prompt, "role": "user"}]
        next_message = dict(content=response, role="assistant")
        return self._template.render(
            prompt_messages=prompt_messages,
            next_message=next_message,
            score=score,
            correct_answer=correct_answer,
            extracted_answer=extracted_answer,
        )
