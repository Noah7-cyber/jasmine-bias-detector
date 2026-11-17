# Visual system: Research workbench

## Direction

Confirmed by the user: clear research workbench for students and auditors. Desktop-first, legible and restrained; the interface helps compare evidence, sample sizes, settings and uncertainty. The metrics and data remain the focus.

## Surface rules

- Warm off-white page; pale sage navigation surface; deep forest ink and one forest action color.
- System sans-serif for interface copy; system monospace only for measured values and code-like identifiers.
- Group data appear in native tables and explicit Plotly rate charts with axes, text values and labels; color never carries the result alone.
- Warning and uncertainty copy appears next to the metric it qualifies. The layout avoids “fair/unfair” verdicts.
- Navigation follows Dataset → Analysis → Results → Report. Keep selections and state visible in the sidebar.
- Use standard Streamlit controls with clear labels, keyboard focus, responsive columns and readable text. No remote font or image requests.
- Uploaded data is not globally cached; only public bundled sample datasets may use shared caching.
- The sidebar owns navigation, with a compact workflow indicator and contextual next-step buttons. Dataset summaries show row, column and missing-value counts; the upload screen includes a minimal CSV example. Captions use the muted ink token at full opacity for readability.
