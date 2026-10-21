from __future__ import annotations
import os
import torch
import gradio as gr
import spaces
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from analysis import (
    token_surprisal,
    surprisal_html,
    greedy_continue,
    contrast_rows,
    md_table,
    WIKI_FEWSHOT,
    CONTRAST_SET,
)

MODEL_REPO = os.environ.get("MODEL_REPO", "OpenPeerAI/My-Name-is-Riemann-Prerelease")
CONTROLLER_PATH = "ntk_controller/stage6_12.85b.pt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
)
model.eval()
N_PARAMS = sum(p.numel() for p in model.parameters())

_ctl = {"tuner": None, "attached": False}


def _controller():
    if _ctl["tuner"] is None:
        from ntkmirror import ForwardFineTuner

        path = hf_hub_download(MODEL_REPO, CONTROLLER_PATH)
        tuner = ForwardFineTuner(model, tokenizer)
        tuner.load(path, allow_model_mismatch=True, map_location="cpu")
        _ctl["tuner"] = tuner
    return _ctl["tuner"]


def _attach(on: bool):
    tuner = _controller()
    if on and not _ctl["attached"]:
        tuner.controller.to(model.device)
        tuner.controller.attach()
        _ctl["attached"] = True
    elif not on and _ctl["attached"]:
        tuner.controller.remove()
        _ctl["attached"] = False


def _gpu():
    if model.device.type != "cuda":
        model.to("cuda")


@spaces.GPU(duration=90)
def complete_wiki(subject, opening, max_new, temperature):
    _gpu()
    _attach(False)
    prompt = WIKI_FEWSHOT + f"{subject.strip()} {opening.strip()}".strip()
    text = greedy_continue(
        model,
        tokenizer,
        prompt,
        int(max_new),
        repetition_penalty=1.25,
        temperature=temperature,
        max_context=2048,
    )
    return f"{subject.strip()} {opening.strip()}".strip() + text


@spaces.GPU(duration=60)
def analyse(text, use_ctl):
    _gpu()
    _attach(False)
    base = token_surprisal(model, tokenizer, text, max_length=2048)
    out = surprisal_html(base, "Base weights:")
    if use_ctl:
        _attach(True)
        ctl = token_surprisal(model, tokenizer, text, max_length=2048)
        _attach(False)
        delta = ctl["nll"] - base["nll"]
        out += "<hr/>" + surprisal_html(ctl, "With NTK controller:")
        out += f"<p><b>Controller effect on this text:</b> mean surprisal {base['nll']:.3f} to {ctl['nll']:.3f} nats ({delta:+.3f}); perplexity {base['ppl']:.1f} to {ctl['ppl']:.1f}.</p>"
    return out


@spaces.GPU(duration=60)
def contrast():
    _gpu()
    _attach(False)
    rows = contrast_rows(model, tokenizer, CONTRAST_SET)
    table = md_table([[r[0], r[1], r[2]] for r in rows])
    best, worst = min(rows, key=lambda r: r[1]), max(rows, key=lambda r: r[1])
    return (
        table
        + f"\n\nThe model assigns the fluent sentence a perplexity of {best[1]} and the random-word string {worst[1]}, "
        f"a ratio of about {worst[1]/best[1]:,.0f}x. That spread is what a language model looks like from the inside: "
        "it has learned which sequences are English and which are not, independent of whether it can chat."
    )


DESCRIPTION = f"""
# My Name is Riemann Prerelease - Basic Demo

**{N_PARAMS/1e9:.1f}B parameters.** Grown rung by rung from
[OpenPeerLLM](https://huggingface.co/OpenPeerAI/OpenPeerLLM) and fine-tuned with the
[OpenPeer-AI-NTK-Trainer](https://huggingface.co/OpenPeerAI/OpenPeer-AI-NTK-Trainer)
gate controller instead of LoRA. Weights: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}).

This is a **prerelease base model**, trained on about 2.2 billion tokens of encyclopedic text, agentic coding
transcripts and source code. It is not instruction-tuned and does not chat. The tabs below show what it has
actually learned: completing text in the register it was trained on, scoring how expected each token of a
passage is, and, most tellingly, telling fluent language from noise.
"""

with gr.Blocks(title="My Name is Riemann Prerelease - Basic Demo") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tab("Complete an article"):
        gr.Markdown(
            "Give a subject and an opening clause in encyclopedia register; the model continues it. Two example articles are prepended as context."
        )
        with gr.Row():
            subject = gr.Textbox(label="Subject", value="The Nile")
            opening = gr.Textbox(
                label="Opening clause",
                value="is a major north-flowing river in northeastern Africa. It",
            )
        with gr.Row():
            max_new = gr.Slider(24, 160, value=80, step=8, label="Max new tokens")
            temp = gr.Slider(
                0.0, 1.0, value=0.0, step=0.1, label="Temperature (0 = greedy)"
            )
        wiki_out = gr.Textbox(label="Completion", lines=8)
        gr.Button("Complete", variant="primary").click(
            complete_wiki, [subject, opening, max_new, temp], wiki_out
        )
        gr.Examples(
            [
                ["Mount Kilimanjaro", "is a dormant volcano in Tanzania. It"],
                [
                    "The Rosetta Stone",
                    "is a stele inscribed with three versions of a decree issued in",
                ],
                ["Alexandria", "is the second-largest city in Egypt and"],
            ],
            inputs=[subject, opening],
        )

    with gr.Tab("Analyse text"):
        gr.Markdown(
            "Paste any text. Each token is coloured by how surprised the model was to see it. Fluent text in a register the model knows comes out green; errors, nonsense and unfamiliar registers come out red. With the controller enabled you also see how much the gates change the model's view of the text."
        )
        a_text = gr.Textbox(
            label="Text",
            lines=5,
            value="The Great Pyramid of Giza is the oldest of the Seven Wonders of the Ancient World, and the only one to remain largely intact.",
        )
        a_ctl = gr.Checkbox(value=True, label="Compare with NTK controller attached")
        a_out = gr.HTML()
        gr.Button("Analyse", variant="primary").click(analyse, [a_text, a_ctl], a_out)

    with gr.Tab("Fluency test"):
        gr.Markdown(
            "Six fixed strings, from fluent encyclopedic English down to random words, scored by the model. A working language model separates these by orders of magnitude."
        )
        c_out = gr.Markdown()
        gr.Button("Run fluency test", variant="primary").click(contrast, None, c_out)

    gr.Markdown(
        "Licensed under the Open Innovation License v2 (OPNL-2) and MIT. "
        "Developed by OpenPeer AI, Riemann Computing Inc., and Andrew Magdy Kamal."
    )

demo.queue(max_size=16).launch()
