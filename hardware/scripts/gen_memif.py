#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 IObundle
#
# SPDX-License-Identifier: MIT

"""Hoist the SpinalHDL-blackboxed VexRiscv memories to py2hwsw memory ports.

The SpinalHDL generator is invoked with
`addStandardMemBlackboxing(blackboxWhatsYouCan)`, which turns every internal
`Mem` of the CPU into a `Ram_1w_1rs` blackbox instance.  Those instances live
inside the generated `VexRiscvAxi4Linux`, `InstructionCache` and `DataCache`
modules.

This script reads the generated Verilog and produces a new file whose top
module (`VexRiscvAxi4LinuxMemIf`) exposes each memory through a set of ports
that match the py2hwsw `ram_t2p` / `ram_t2p_be` manager interface.  The
py2hwsw `iob_memwrapper` can then instantiate the actual `iob_ram_*` memories
and forward the `MEM_ASIC` parameter, so the same RTL works for simulation,
FPGA and ASIC flows.

Only the memory instances are removed from the RTL; the rest of the CPU is
left untouched.  The script is deterministic and is re-run after every VexRiscv
regeneration (`make vexriscv`).
"""

import argparse
import re
import sys

# (module, instance_name) -> (port prefix, py2hwsw memory interface type)
MEMORIES = {
    ("VexRiscvAxi4Linux", "RegFilePlugin_regFile"): ("regfile_a_", "ram_t2p"),
    ("VexRiscvAxi4Linux", "RegFilePlugin_regFile_1"): ("regfile_b_", "ram_t2p"),
    ("InstructionCache", "banks_0"): ("icache_data_", "ram_t2p"),
    ("InstructionCache", "ways_0_tags"): ("icache_tags_", "ram_t2p"),
    ("DataCache", "ways_0_tags"): ("dcache_tags_", "ram_t2p"),
    ("DataCache", "ways_0_data"): ("dcache_data_", "ram_t2p_be"),
}

TOP_MODULE = "VexRiscvAxi4Linux"
OUT_MODULE = "VexRiscvAxi4LinuxMemIf"

# Instances in the top module whose memory ports are passed through.
# instance_name -> module name
PASS_THROUGH_INSTANCES = {
    "IBusCachedPlugin_cache": "InstructionCache",
    "dataCache_1": "DataCache",
}

MODULE_RE = re.compile(r"\bmodule\s+(\w+)\s*\((.*?)\);(.*?)\bendmodule", re.DOTALL)
INSTANCE_RE = re.compile(r"Ram_1w_1rs\s*#\s*\((.*?)\)\s*(\w+)\s*\((.*?)\);", re.DOTALL)
CONN_RE = re.compile(r"\.(\w+)\s*\(\s*([^()]*?)\s*\)")


def parse_instance(match):
    """Return (params, connections) dicts for one Ram_1w_1rs instance."""
    params = {k: v.strip() for k, v in CONN_RE.findall(match.group(1))}
    conns = {k: v.strip() for k, v in CONN_RE.findall(match.group(3))}
    return params, conns


def as_int(value, default):
    """Parse an integer Verilog literal (e.g. ``5`` or ``1'b0``)."""
    if value is None:
        return default
    value = value.strip()
    m = re.fullmatch(r"(\d+)", value)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"\d+'([bdh])([0-9a-fA-F_]+)", value)
    if m:
        base = {"b": 2, "d": 10, "h": 16}[m.group(1)]
        return int(m.group(2).replace("_", ""), base)
    return default


def memory_widths(params):
    """Return (addr_w, data_w, strb_w) for the instance."""
    addr_w = as_int(params.get("wrAddressWidth"), 32)
    data_w = as_int(params.get("wrDataWidth"), 32)
    strb_w = as_int(params.get("wrMaskWidth"), max(1, data_w // 8))
    return addr_w, data_w, strb_w


def port_declaration(kind, name, width=None):
    """Format a Verilog port declaration."""
    if width is None or width <= 1:
        return f"  {kind} wire          {name}"
    return f"  {kind} wire [{width - 1}:0] {name}"


def memory_ports(prefix, mem_type, addr_w, data_w, strb_w):
    """Return the port declarations for one memory manager interface."""
    ports = [port_declaration("output", f"{prefix}clk_o")]
    if mem_type == "ram_t2p_be":
        ports.append(port_declaration("output", f"{prefix}w_strb_o", strb_w))
    else:
        ports.append(port_declaration("output", f"{prefix}w_en_o"))
    ports += [
        port_declaration("output", f"{prefix}w_addr_o", addr_w),
        port_declaration("output", f"{prefix}w_data_o", data_w),
        port_declaration("output", f"{prefix}r_en_o"),
        port_declaration("output", f"{prefix}r_addr_o", addr_w),
        port_declaration("input", f"{prefix}r_data_i", data_w),
    ]
    return ports


def signal_names(prefix, mem_type):
    """Return the py2hwsw manager interface signal names for a memory."""
    names = [f"{prefix}clk_o"]
    names.append(f"{prefix}w_strb_o" if mem_type == "ram_t2p_be" else f"{prefix}w_en_o")
    names += [
        f"{prefix}w_addr_o",
        f"{prefix}w_data_o",
        f"{prefix}r_en_o",
        f"{prefix}r_addr_o",
        f"{prefix}r_data_i",
    ]
    return names


def memory_assigns(prefix, mem_type, conns, strb_w):
    """Return assign statements replacing a Ram_1w_1rs instance.

    The SpinalHDL ``Ram_1w_1rs`` primitive only writes a byte lane when both
    ``wr_en`` and the corresponding ``wr_mask`` bit are set, while the py2hwsw
    ``ram_t2p_be`` interface only carries the byte strobes.  Fold ``wr_en`` into
    the strobes so the write semantics are preserved.

    Likewise, the primitive gates the read register with ``rd_en`` only
    (``rd_dataEn`` does not gate it), so ``r_en`` must not be combined with
    ``rd_dataEn``.
    """
    lines = [f"  assign {prefix}clk_o = {conns['wr_clk']};"]
    if mem_type == "ram_t2p_be":
        lines.append(
            f"  assign {prefix}w_strb_o = {conns['wr_en']} ? "
            f"{conns['wr_mask']} : {strb_w}'b0;"
        )
    else:
        lines.append(f"  assign {prefix}w_en_o = {conns['wr_en']};")
    lines += [
        f"  assign {prefix}w_addr_o = {conns['wr_addr']};",
        f"  assign {prefix}w_data_o = {conns['wr_data']};",
        f"  assign {prefix}r_en_o = {conns['rd_en']};",
        f"  assign {prefix}r_addr_o = {conns['rd_addr']};",
        f"  assign {conns['rd_data']} = {prefix}r_data_i;",
    ]
    return "\n".join(lines)


def find_instance_span(text, instance_name, module_name):
    """Find the character span of `<module> <instance> ( ... );`."""
    pattern = re.compile(
        r"\b" + re.escape(module_name) + r"\s+" + re.escape(instance_name) + r"\s*\("
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"Instance {module_name} {instance_name} not found")
    start = match.start()
    depth = 0
    i = match.end() - 1
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                end = text.find(";", i)
                if end == -1:
                    raise RuntimeError(f"Unterminated instance {instance_name}")
                return start, end + 1
        i += 1
    raise RuntimeError(f"Unbalanced parentheses in instance {instance_name}")


def transform(text):
    """Transform the generated Verilog and return the MemIf version."""
    modules = list(MODULE_RE.finditer(text))
    if not modules:
        raise RuntimeError("No modules found")

    # First pass: locate every memory instance, record its parameters and the
    # replacement text, and build the port list for each owning module.
    memory_params = {}
    module_ports = {}
    replacements = []
    for module_match in modules:
        module_name = module_match.group(1)
        body = module_match.group(3)
        body_start = module_match.start(3)
        ports = []
        for inst_match in INSTANCE_RE.finditer(body):
            key = (module_name, inst_match.group(2))
            if key not in MEMORIES:
                continue
            prefix, mem_type = MEMORIES[key]
            params, conns = parse_instance(inst_match)
            memory_params[key] = params
            addr_w, data_w, strb_w = memory_widths(params)
            ports += memory_ports(prefix, mem_type, addr_w, data_w, strb_w)
            replacements.append(
                (
                    body_start + inst_match.start(),
                    body_start + inst_match.end(),
                    # The instance's original indentation is kept, so drop the
                    # indent from the first assign line.
                    memory_assigns(prefix, mem_type, conns, strb_w).lstrip(),
                )
            )
        if ports:
            module_ports[module_name] = ports

    missing = {f"{m}.{i}" for m, i in MEMORIES} - {f"{m}.{i}" for m, i in memory_params}
    if missing:
        found = sorted(f"{m}.{i}" for m, i in memory_params)
        raise RuntimeError(
            f"Could not locate all memories. Missing: {sorted(missing)}. "
            f"Found: {found}"
        )

    # The top module also exposes the memory ports of its cache submodules.
    top_ports = module_ports.get(TOP_MODULE, [])
    for module_name in PASS_THROUGH_INSTANCES.values():
        for (mod, inst), (prefix, mem_type) in MEMORIES.items():
            if mod != module_name:
                continue
            addr_w, data_w, strb_w = memory_widths(memory_params[(mod, inst)])
            top_ports += memory_ports(prefix, mem_type, addr_w, data_w, strb_w)
    if top_ports:
        module_ports[TOP_MODULE] = top_ports

    # Replace each instance with assign statements (from the end).
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]

    # Re-parse and add the precomputed ports to each module header.
    modules = list(MODULE_RE.finditer(text))
    out = []
    last = 0
    for module_match in modules:
        module_name = module_match.group(1)
        module_text = module_match.group(0)

        ports = module_ports.get(module_name, [])

        if ports:
            open_idx = module_text.index("(")
            close_idx = module_text.index(");", open_idx)
            module_text = (
                module_text[:open_idx]
                + module_text[open_idx:close_idx].rstrip()
                + ",\n"
                + ",\n".join(ports)
                + "\n"
                + module_text[close_idx:]
            )

        if module_name == TOP_MODULE:
            module_text = module_text.replace(
                f"module {TOP_MODULE} (", f"module {OUT_MODULE} (", 1
            )
            module_text = module_text.replace(
                f"// Component : {TOP_MODULE}", f"// Component : {OUT_MODULE}", 1
            )

        out.append(text[last : module_match.start()])
        out.append(module_text)
        last = module_match.end()
    out.append(text[last:])
    text = "".join(out)

    # Pass the submodule memory ports through the top-level cache instances.
    for instance_name, module_name in PASS_THROUGH_INSTANCES.items():
        start, end = find_instance_span(text, instance_name, module_name)
        block = text[start:end]
        conns = []
        for (mod, inst), (prefix, mem_type) in MEMORIES.items():
            if mod != module_name:
                continue
            conns += [f"    .{name}({name})" for name in signal_names(prefix, mem_type)]
        if conns:
            close_idx = block.rindex(");")
            block = (
                block[:close_idx]
                + ",\n"
                + ",\n".join(conns)
                + "\n  "
                + block[close_idx:]
            )
            text = text[:start] + block + text[end:]

    marker = (
        "// This file is generated by hardware/scripts/gen_memif.py from the\n"
        "// SpinalHDL output VexRiscvAxi4Linux.v. Do not edit it manually.\n"
        "// The VexRiscv internal memories are exposed as py2hwsw memory\n"
        "// interfaces so that iob_memwrapper instantiates them.\n\n"
    )
    return marker + text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to the generated VexRiscvAxi4Linux.v")
    parser.add_argument("output", help="Path to write VexRiscvAxi4LinuxMemIf.v")
    args = parser.parse_args()

    with open(args.input) as f:
        text = f.read()

    try:
        result = transform(text)
    except Exception as exc:  # noqa: BLE001
        print(f"gen_memif: error: {exc}", file=sys.stderr)
        return 1

    with open(args.output, "w") as f:
        f.write(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
