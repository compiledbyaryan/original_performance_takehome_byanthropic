"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name, length=VLEN)

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def pack_instructions(self, raw_slots):
        packed_instrs = []
        current_bundle = defaultdict(list)
        current_defs = set()
        
        def get_rw_sets(engine, slot):
            reads = set()
            writes = set()
            op = slot[0]
            args = slot[1:] 

            if engine == 'alu': 
                writes.add(args[0])
                if len(args) > 1: reads.add(args[1])
                if len(args) > 2: reads.add(args[2])
            
            elif engine == 'valu':
                if op == 'vbroadcast': 
                    for k in range(VLEN): writes.add(args[0] + k)
                    reads.add(args[1])
                elif op == 'multiply_add':
                    for k in range(VLEN):
                        writes.add(args[0] + k)
                        reads.add(args[1] + k)
                        reads.add(args[2] + k)
                        reads.add(args[3] + k)
                else: 
                    for k in range(VLEN):
                        writes.add(args[0] + k)
                        if len(args) > 1: reads.add(args[1] + k)
                        if len(args) > 2: reads.add(args[2] + k)

            elif engine == 'load':
                if op == 'load': 
                    writes.add(args[0])
                    reads.add(args[1])
                elif op == 'vload': 
                    for k in range(VLEN): writes.add(args[0] + k)
                    reads.add(args[1])
                elif op == 'const': 
                    writes.add(args[0])
                elif op == 'load_offset': 
                    writes.add(args[0])
                    reads.add(args[1])

            elif engine == 'store':
                reads.add(args[0])
                if op == 'store': reads.add(args[1])
                elif op == 'vstore':
                    for k in range(VLEN): reads.add(args[1] + k)

            elif engine == 'flow':
                if op == 'select': 
                    writes.add(args[0])
                    reads.add(args[1])
                    reads.add(args[2])
                    reads.add(args[3])
                elif op == 'add_imm': 
                    writes.add(args[0])
                    reads.add(args[1])
                elif op == 'coreid': 
                    writes.add(args[0])
                elif op == 'vselect':
                    for k in range(VLEN): 
                        writes.add(args[0] + k)
                        reads.add(args[1] + k)
                        reads.add(args[2] + k)
                        reads.add(args[3] + k)
                elif op == 'cond_jump':
                    reads.add(args[0])
            
            return reads, writes

        for engine, slot in raw_slots:
            reads, writes = get_rw_sets(engine, slot)
            
            if len(current_bundle[engine]) >= SLOT_LIMITS[engine]:
                packed_instrs.append(dict(current_bundle))
                current_bundle = defaultdict(list)
                current_defs = set()

            if not reads.isdisjoint(current_defs):
                packed_instrs.append(dict(current_bundle))
                current_bundle = defaultdict(list)
                current_defs = set()
            
            current_bundle[engine].append(slot)
            current_defs.update(writes)

        if current_bundle:
            packed_instrs.append(dict(current_bundle))
            
        return packed_instrs

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        self.instrs = [] 
        raw_ops = []

        def emit(engine, slot):
            raw_ops.append((engine, slot))

        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars: self.alloc_scratch(v, 1)

        idx_vec = self.alloc_vec("idx_vec")
        val_vec = self.alloc_vec("val_vec")
        node_val_vec = self.alloc_vec("node_val_vec")
        tmp_vec1 = self.alloc_vec("tmp_vec1")
        tmp_vec2 = self.alloc_vec("tmp_vec2")
        addr_calc_vec = self.alloc_vec("addr_calc_vec")
        
        const_vec_0 = self.alloc_vec("const_0")
        const_vec_1 = self.alloc_vec("const_1")
        const_vec_2 = self.alloc_vec("const_2")
        const_vec_n_nodes = self.alloc_vec("const_n_nodes")
        
        hash_consts = []
        for stage in HASH_STAGES:
            c1 = self.alloc_vec(f"h_c1_{stage[1]}")
            c3 = self.alloc_vec(f"h_c3_{stage[4]}")
            hash_consts.append((c1, stage[1], c3, stage[4]))

        # --- Initialization ---
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        for i, v in enumerate(init_vars):
            emit("load", ("const", tmp_scalar, i))
            emit("load", ("load", self.scratch[v], tmp_scalar))

        emit("load", ("const", tmp_scalar, 0))
        emit("valu", ("vbroadcast", const_vec_0, tmp_scalar))
        emit("load", ("const", tmp_scalar, 1))
        emit("valu", ("vbroadcast", const_vec_1, tmp_scalar))
        emit("load", ("const", tmp_scalar, 2))
        emit("valu", ("vbroadcast", const_vec_2, tmp_scalar))
        emit("load", ("load", tmp_scalar, self.scratch["n_nodes"]))
        emit("valu", ("vbroadcast", const_vec_n_nodes, tmp_scalar))

        for (c1_addr, v1, c3_addr, v3) in hash_consts:
             emit("load", ("const", tmp_scalar, v1))
             emit("valu", ("vbroadcast", c1_addr, tmp_scalar))
             emit("load", ("const", tmp_scalar, v3))
             emit("valu", ("vbroadcast", c3_addr, tmp_scalar))
        
        emit("flow", ("pause",))

        # --- Main Loop ---
        ptr_indices = self.scratch["inp_indices_p"]
        ptr_values = self.scratch["inp_values_p"]
        ptr_forest = self.scratch["forest_values_p"]
        
        curr_indices_ptr = self.alloc_scratch("curr_idx_ptr")
        curr_values_ptr = self.alloc_scratch("curr_val_ptr")
        forest_base_vec = self.alloc_vec("forest_base_vec")
        
        # CORRECTED: Broadcast the address, do not load from it
        emit("valu", ("vbroadcast", forest_base_vec, ptr_forest))

        for r in range(rounds):
            # CORRECTED: Copy base pointers using add_imm + 0, do not load (dereference)
            emit("flow", ("add_imm", curr_indices_ptr, ptr_indices, 0))
            emit("flow", ("add_imm", curr_values_ptr, ptr_values, 0))

            for b in range(0, batch_size, VLEN):
                emit("load", ("vload", idx_vec, curr_indices_ptr))
                emit("load", ("vload", val_vec, curr_values_ptr))

                emit("valu", ("+", addr_calc_vec, forest_base_vec, idx_vec))
                for k in range(VLEN):
                    emit("load", ("load", node_val_vec + k, addr_calc_vec + k))

                emit("valu", ("^", val_vec, val_vec, node_val_vec))
                
                for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    c1_vec, _, c3_vec, _ = hash_consts[i]
                    emit("valu", (op1, tmp_vec1, val_vec, c1_vec))
                    emit("valu", (op3, tmp_vec2, val_vec, c3_vec))
                    emit("valu", (op2, val_vec, tmp_vec1, tmp_vec2))

                emit("valu", ("&", tmp_vec1, val_vec, const_vec_1)) 
                emit("valu", ("==", tmp_vec1, tmp_vec1, const_vec_0))
                
                emit("flow", ("vselect", tmp_vec2, tmp_vec1, const_vec_1, const_vec_2))
                
                emit("valu", ("<<", idx_vec, idx_vec, const_vec_1))
                emit("valu", ("+", idx_vec, idx_vec, tmp_vec2))
                
                emit("valu", ("<", tmp_vec1, idx_vec, const_vec_n_nodes))
                emit("flow", ("vselect", idx_vec, tmp_vec1, idx_vec, const_vec_0))

                emit("store", ("vstore", curr_indices_ptr, idx_vec))
                emit("store", ("vstore", curr_values_ptr, val_vec))

                emit("flow", ("add_imm", curr_indices_ptr, curr_indices_ptr, VLEN))
                emit("flow", ("add_imm", curr_values_ptr, curr_values_ptr, VLEN))

            emit("flow", ("pause",))

        self.instrs = self.pack_instructions(raw_ops)
BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
