import os

from fatigue.io import configs

# 1. Create a dummy YAML file
dummy_yaml = """
project:
  name: "test_project"
  version: 1.0
training:
  lr: 0.01
"""
with open("temp_test.yaml", "w") as f:
    f.write(dummy_yaml)

print("--- Testing Parse ---")
# 2. Test Parsing
cfg = configs.parse_file("temp_test.yaml")
print(f"Project Name: {cfg.project.name}")  # Should print "test_project"

print("\n--- Testing Overrides ---")
# 3. Test Merging/Overrides (Simulating CLI arguments)
override_string = "training.lr=0.05"
override_cfg = configs.parse_string(override_string)

final_cfg = configs.merge_configs([cfg, override_cfg])
print(f"Original LR: {cfg.training.lr}")  # Should be 0.01
print(f"Final LR:    {final_cfg.training.lr}")  # Should be 0.05

# Cleanup
os.remove("temp_test.yaml")
print("\nSuccess! config.py is working.")
