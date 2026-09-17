"""Exact per-segment prompt token counts, using the served model's own tokenizer and chat template."""

from bisect import bisect_right
from pathlib import Path

from .tasks import SEGMENTS, to_messages

TEMPLATES = Path(__file__).parent / "templates"
# Llama 3.1 templates print today's date into the prompt; pin it so prompts are identical on every day.
PIN_DATE = {"date_string": "26 Jul 2024"}

MODELS = {
    "qwen2.5-7b": {"hf": "Qwen/Qwen2.5-7B-Instruct-AWQ", "parser": "hermes"},
    "llama3.1-8b": {"hf": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", "parser": "llama3_json",
                    "template": "tool_chat_template_llama3.1_json.jinja", "template_kwargs": PIN_DATE},
    "qwen2.5-0.5b": {"hf": "Qwen/Qwen2.5-0.5B-Instruct", "parser": "hermes"},  # smoke tests
}


class Prompter:
    """Renders chat requests the way vLLM does and attributes every prompt token to a segment."""

    def __init__(self, tokenizer_id, template_file=None, template_kwargs=None):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(tokenizer_id)
        self.template = (TEMPLATES / template_file).read_text(encoding="utf-8") if template_file else None
        self.kwargs = template_kwargs or {}

    @classmethod
    def for_model(cls, model, tokenizer=None):
        """Prompter for a preset (or an explicit HF tokenizer id); None when neither is known."""
        preset = MODELS.get(model, {})
        tokenizer = tokenizer or preset.get("hf")
        return cls(tokenizer, preset.get("template"), preset.get("template_kwargs")) if tokenizer else None

    def render(self, messages, tools) -> str:
        return self.tok.apply_chat_template(messages, tools=tools, chat_template=self.template,
                                            add_generation_prompt=True, tokenize=False, **self.kwargs)

    def count(self, text) -> int:
        # vLLM tokenizes rendered chat prompts without adding special tokens again.
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def attribute(self, msgs, tools) -> dict:
        """Token count per segment (+ query, template, total) for a layout() result."""
        pieces = [piece for _, parts in msgs for piece in parts]
        marks = [f"{i}" for i in range(len(pieces))]
        marked, k = [], 0
        for role, parts in msgs:
            marked.append({"role": role, "content": "".join(marks[k:k + len(parts)])})
            k += len(parts)
        skeleton = self.render(marked, tools)
        if any(skeleton.count(m) != 1 for m in marks):
            raise ValueError("chat template dropped or duplicated message content")

        # Put the real text back where each marker landed, remembering its character span.
        prompt, spans, pos = [], [], 0
        for at, i in sorted((skeleton.index(m), i) for i, m in enumerate(marks)):
            prompt.append(skeleton[pos:at])
            start = sum(map(len, prompt))
            seg, text = pieces[i]
            spans.append((start, start + len(text), seg))
            prompt.append(text)
            pos = at + len(marks[i])
        prompt.append(skeleton[pos:])
        prompt = "".join(prompt)
        messages = to_messages(msgs)
        if prompt != self.render(messages, tools):
            raise ValueError("chat template rewrites message content; attribution would be inexact")

        enc = self.tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        starts = [s for s, _, _ in spans]
        counts = dict.fromkeys((*SEGMENTS, "query", "template"), 0)
        for tok_start, _ in enc["offset_mapping"]:
            j = bisect_right(starts, tok_start) - 1
            counts[spans[j][2] if j >= 0 and tok_start < spans[j][1] else "template"] += 1
        total = len(enc["input_ids"])
        if tools:
            # Tool schemas are rendered by the template itself: charge the full marginal cost to `tools`.
            counts["tools"] = total - self.count(self.render(messages, None))
            counts["template"] -= counts["tools"]
        counts["total"] = total
        return counts
