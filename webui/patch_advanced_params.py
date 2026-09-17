#!/usr/bin/env python3
"""Insert Qwen thinking controls into Open WebUI AdvancedParams.svelte."""

from __future__ import annotations

import sys
from pathlib import Path

QWEN_DEFAULTS = "\t\tenable_thinking: null,\n\t\tpreserve_thinking: null,\n"

REASONING_OLD = """				<button
					class="p-1 px-3 text-xs flex rounded-sm transition shrink-0 outline-hidden"
					type="button"
					on:click={() => {
						params.reasoning_effort = (params?.reasoning_effort ?? null) === null ? 'medium' : null;
					}}
				>
					{#if (params?.reasoning_effort ?? null) === null}
						<span class="ml-2 self-center"> {$i18n.t('Default')} </span>
					{:else}
						<span class="ml-2 self-center"> {$i18n.t('Custom')} </span>
					{/if}
				</button>
			</div>
		</Tooltip>

		{#if (params?.reasoning_effort ?? null) !== null}
			<div class="flex mt-0.5 space-x-2">
				<div class=" flex-1">
					<input
						class="text-sm w-full bg-transparent outline-hidden outline-none"
						type="text"
						aria-label={$i18n.t('Reasoning Effort')}
						placeholder={$i18n.t('Enter reasoning effort')}
						bind:value={params.reasoning_effort}
						autocomplete="off"
					/>
				</div>
			</div>
		{/if}
	</div>
"""

REASONING_NEW = """				<button
					class="p-1 px-3 text-xs flex rounded-sm transition shrink-0 outline-hidden"
					type="button"
					on:click={() => {
						const cur = params?.reasoning_effort ?? null;
						if (cur === null) {
							params.reasoning_effort = 'xhigh';
						} else if (cur === 'xhigh') {
							params.reasoning_effort = 'medium';
						} else if (cur === 'medium') {
							params.reasoning_effort = 'low';
						} else {
							params.reasoning_effort = null;
						}
					}}
				>
					{#if (params?.reasoning_effort ?? null) === null}
						<span class="ml-2 self-center"> {$i18n.t('Default')} </span>
					{:else}
						<span class="ml-2 self-center"> {params.reasoning_effort} </span>
					{/if}
				</button>
			</div>
		</Tooltip>
	</div>
"""

QWEN_BLOCKS = """
	<div class=" py-0.5 w-full justify-between">
		<Tooltip
			content="Qwen chat_template_kwargs.enable_thinking. Off skips the thinking phase."
			placement="top-start"
			className="inline-tooltip"
		>
			<div class="flex w-full justify-between">
				<div class=" self-center text-xs">
					{'enable_thinking'} (Qwen)
				</div>
				<button
					class="p-1 px-3 text-xs flex rounded-sm transition shrink-0 outline-hidden"
					type="button"
					on:click={() => {
						if ((params?.enable_thinking ?? null) === null) {
							params.enable_thinking = true;
						} else if (params.enable_thinking === true) {
							params.enable_thinking = false;
						} else {
							params.enable_thinking = null;
						}
					}}
				>
					{#if params.enable_thinking === true}
						<span class="ml-2 self-center"> {$i18n.t('On')} </span>
					{:else if params.enable_thinking === false}
						<span class="ml-2 self-center"> {$i18n.t('Off')} </span>
					{:else}
						<span class="ml-2 self-center"> {$i18n.t('Default')} </span>
					{/if}
				</button>
			</div>
		</Tooltip>
	</div>

	<div class=" py-0.5 w-full justify-between">
		<Tooltip
			content="Qwen chat_template_kwargs.preserve_thinking. Keep prior thinking in multi-turn chats."
			placement="top-start"
			className="inline-tooltip"
		>
			<div class="flex w-full justify-between">
				<div class=" self-center text-xs">
					{'preserve_thinking'} (Qwen)
				</div>
				<button
					class="p-1 px-3 text-xs flex rounded-sm transition shrink-0 outline-hidden"
					type="button"
					on:click={() => {
						if ((params?.preserve_thinking ?? null) === null) {
							params.preserve_thinking = true;
						} else if (params.preserve_thinking === true) {
							params.preserve_thinking = false;
						} else {
							params.preserve_thinking = null;
						}
					}}
				>
					{#if params.preserve_thinking === true}
						<span class="ml-2 self-center"> {$i18n.t('On')} </span>
					{:else if params.preserve_thinking === false}
						<span class="ml-2 self-center"> {$i18n.t('Off')} </span>
					{:else}
						<span class="ml-2 self-center"> {$i18n.t('Default')} </span>
					{/if}
				</button>
			</div>
		</Tooltip>
	</div>

"""

LOGIT_BIAS_START = """	<div class=" py-0.5 w-full justify-between">
		<Tooltip
			content={$i18n.t(
				'Boosting or penalizing specific tokens for constrained responses. Bias values will be clamped between -100 and 100 (inclusive). (Default: none)'
			)}
"""


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "enable_thinking: null" in text and "preserve_thinking: null" in text:
        print(f"already patched: {path}")
        return

    if "\t\treasoning_effort: null,\n" not in text:
        raise SystemExit(f"could not find defaultParams.reasoning_effort in {path}")
    text = text.replace(
        "\t\treasoning_effort: null,\n",
        "\t\treasoning_effort: null,\n" + QWEN_DEFAULTS,
        1,
    )

    if REASONING_OLD not in text:
        raise SystemExit(f"could not find reasoning_effort UI block in {path}")
    text = text.replace(REASONING_OLD, REASONING_NEW, 1)

    if LOGIT_BIAS_START not in text:
        raise SystemExit(f"could not find logit_bias insertion point in {path}")
    text = text.replace(LOGIT_BIAS_START, QWEN_BLOCKS + LOGIT_BIAS_START, 1)

    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "AdvancedParams.svelte")
    patch(target)
