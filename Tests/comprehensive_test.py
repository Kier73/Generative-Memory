import os
import sys
import time
import random
import unittest
import numpy as np

# Ensure gmem is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gmem.core import GMemContext
from gmem.hashing import vl_mask, vl_inverse_mask, MASK64
from gmem.morph import MorphMode

class TestGMemComprehensive(unittest.TestCase):
    def setUp(self):
        self.seed = 0xDEADBEEFCAFEBABE
        self.ctx = GMemContext(self.seed)

    def test_identity_and_determinism(self):
        """Verify that fetching the same address multiple times returns the same value."""
        addr = 1337
        val1 = self.ctx.fetch(addr)
        val2 = self.ctx.fetch(addr)
        self.assertEqual(val1, val2, "Fetch is not deterministic")
        
        # Test different seed
        ctx2 = GMemContext(self.seed + 1)
        val3 = ctx2.fetch(addr)
        self.assertNotEqual(val1, val3, "Different seeds produced the same value")

    def test_sparse_overlay(self):
        """Verify that writing to an address overrides the synthetic value."""
        addr = 42
        original_val = self.ctx.fetch(addr)
        new_val = 3.14159
        
        self.ctx.write(addr, new_val)
        current_val = self.ctx.fetch(addr)
        
        self.assertEqual(current_val, new_val, "Overlay write failed")
        self.assertNotEqual(original_val, new_val, "Original value was same as new value")
        self.assertEqual(self.ctx.overlay_count, 1, "Overlay count incorrect")

    def test_persistence_aof(self):
        """Test AOF persistence: write, detach, re-attach, verify."""
        aof_path = "test_persistence.aof"
        if os.path.exists(aof_path):
            os.remove(aof_path)
            
        try:
            # 1. Attach and write
            self.ctx.persistence_attach(aof_path)
            addr = 999
            test_val = 123.456
            self.ctx.write(addr, test_val)
            self.ctx.persistence_detach()
            
            # 2. Create new context and attach (should replay)
            new_ctx = GMemContext(self.seed)
            new_ctx.persistence_attach(aof_path)
            
            recovered_val = new_ctx.fetch(addr)
            self.assertAlmostEqual(recovered_val, test_val, places=5, msg="AOF recovery failed")
            new_ctx.persistence_detach()
            
        finally:
            self.ctx.persistence_detach() # Ensure original is detached too
            if os.path.exists(aof_path):
                os.remove(aof_path)

    def test_bulk_fetch_and_zmask(self):
        """Validate bulk fetch and Z-Mask logic."""
        start_addr = 1000
        count = 10
        
        # Initial bulk fetch
        vals_initial = self.ctx.fetch_bulk(start_addr, count)
        self.assertEqual(len(vals_initial), count)
        
        # Write to middle of range
        mid_addr = start_addr + 5
        write_val = 777.777
        self.ctx.write(mid_addr, write_val)
        
        # Bulk fetch again
        vals_after = self.ctx.fetch_bulk(start_addr, count)
        self.assertEqual(vals_after[5], write_val, "Bulk fetch failed to pick up overlay write")
        self.assertEqual(vals_after[0], vals_initial[0], "Bulk fetch corrupted clean values")

    def test_mirroring_and_morphing(self):
        """Verify mirroring and morphing."""
        # 1. Mirroring
        source_ctx = GMemContext(0x1)
        target_ctx = GMemContext(0x2)
        
        addr = 10
        source_val = source_ctx.fetch(addr)
        
        target_ctx.mirror_attach(source_ctx)
        self.assertEqual(target_ctx.fetch(addr), source_val, "Mirroring failed")
        target_ctx.mirror_detach()
        
        # 2. Morphing (Linear transform: y = 2x + 1)
        morph_ctx = GMemContext(0x3)
        morph_ctx.morph_attach(source_ctx, mode=1, a=2.0, b=1.0) # mode 1 = linear
        
        expected_morph = (source_val * 2.0) + 1.0
        self.assertAlmostEqual(morph_ctx.fetch(addr), expected_morph, places=5, msg="Morphing failed")

    def test_numerical_invariance(self):
        """Round-trip check for invertible Feistel network (vl_mask)."""
        seeds = [0, 1, 0xABCDEF, 0xFFFFFFFFFFFFFFFF]
        test_values = [0, 1, 0x1234, 0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF]
        
        for s in seeds:
            for x in test_values:
                masked = vl_mask(x, s)
                unmasked = vl_inverse_mask(masked, s)
                self.assertEqual(x, unmasked, f"vl_mask round-trip failed for x={x}, s={s}")

if __name__ == "__main__":
    unittest.main()
