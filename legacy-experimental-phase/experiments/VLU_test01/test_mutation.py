from evolutionary_vnn import VNNGenome

print("🧪 Testing mutation robustness (100,000 iterations)...")
g = VNNGenome()

for i in range(100000):
    g.mutate(1.0)  # 100% mutation rate
    if (i + 1) % 10000 == 0:
        print(f"   {i+1:,} mutations completed...")

print("✅ All mutations successful! No crashes.")
print(f"\nFinal genome sample:")
print(f"  Anchor Inertia: {g.anchor_inertia:.4f}")
print(f"  Hunter Volume Threshold: {g.hunter_volume_threshold:.4f}")
print(f"  Trauma Multiplier: {g.trauma_radius_multiplier:.4f}")
