---
title: My Name is Riemann Prerelease - Basic Demo
emoji: 🔥
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: other
license_name: opnl-2-and-mit
license_link: https://huggingface.co/OpenPeerAI/My-Name-is-Riemann-Prerelease/blob/main/LICENSE
models:
  - OpenPeerAI/My-Name-is-Riemann-Prerelease
  - OpenPeerAI/OpenPeerLLM
short_description: Chat with the 13.1B My Name is Riemann Prerelease on ZeroGPU
---

# My Name is Riemann Prerelease - Basic Demo

Interactive text generation with the stage 6 checkpoint of
[My Name is Riemann](https://huggingface.co/OpenPeerAI/My-Name-is-Riemann-Prerelease),
a 13.1 billion parameter model grown from OpenPeerLLM's published weights and
fine-tuned with the OpenPeer-AI-NTK-Trainer gate controller. The toggle in the
sidebar attaches or removes the controller so the two can be compared on the
same prompt.

The model loads in bf16 and runs on a ZeroGPU slot. Expect a cold start of a
minute or two the first time a GPU is assigned.

Developed by OpenPeer AI, Riemann Computing Inc., and Andrew Magdy Kamal.
Released under the Open Innovation License v2 and MIT.
