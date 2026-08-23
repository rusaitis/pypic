# Codegen

`pypic.codegen` exports pypic's canonical tables — field aliases, the
recipe registry, per-species templates, and field metadata — as JSON, so
the sibling tools in the [ecosystem](../index.md#ecosystem) can generate
the same names and units without reimplementing them.

The tables are the authority; this module is a serializer over them. A
quantity added with
[`register_recipe`][pypic.compute.register_recipe] shows up in the
bundle automatically.

Each function is also a CLI subcommand — `pypic export bundle`,
`pypic export aliases`, `pypic export recipes`, `pypic export fields`.
See [Command Line](cli.md#schema-and-codegen).

::: pypic.codegen
    options:
      show_root_heading: false
      members_order: source
