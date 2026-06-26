Prompt Assembly Paths available in the Control Panel as of 5/12/2026

In order of control-flow in the make_cluster_analysis_system_prompt function:

Route 1 — Escape Hatch (template_path / template_string)
Bypasses everything. Raw template + appended screen context. No registry, no steps.

Route 2 — Flexible Component-Order (component_order)
Arbitrary ordering of shorthand keys via assemble_from_component_order(). Keys from COMPONENT_REGISTRY. Supports component_overrides to swap wording. Can be numbered (cot_mode=True) or flat. Available in the function signature but not surfaced in notebook cells---benchmark functionality only for now.

Route 3 — Mode-Based Defaults (what the notebook currently uses)
| Route   | MODE     | USE_MCP | Components Assembled                                                          | Request Type                   |
|---------|----------|---------|-------------------------------------------------------------------------------|--------------------------------|
| 3a      | standard | False   | CAT → SC → GCR → NPR → UPR → PCC → O (7)                                    | Flat, single call              |
| 3a+mcp  | standard | True    | CAT → SC → GCR → NPR → UPR → PCC → LIT → O (8)                              | Flat, single call w/ PubMed    |
| 3b      | cot      | False   | CAT → SC → cPH → cGCR → cPri → cPSC → cVer → cO (8 numbered)               | Single call                    |
| 3b+mcp  | cot      | True    | CAT → SC → cPH → cGCR → cPri → LIT → cPSC → cVer → cO (9 numbered)         | Single call w/ PubMed          |
| 3c      | stepwise | False   | System: CAT+SC; Turns: cPH → cGCR → cPri → cPSC → cVer → cO (6 turns)       | Multi-turn                     |
| 3c+mcp  | stepwise | True    | System: CAT+SC; Turns: cPH → cGCR → cPri → LIT(mcp) → cPSC → cVer → cO (7) | Multi-turn, MCP on lit turn    |

Key difference between standard and CoT: standard uses baseline components directly
(GCR, NPR, UPR, PCC, O as separate blocks), while CoT uses wrapped versions
(cGCR embeds GCR, cPri embeds NPR+UPR, cPSC embeds PCC, cO embeds O)
