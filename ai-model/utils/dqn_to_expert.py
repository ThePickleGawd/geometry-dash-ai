import torch
from model import DQNModel, ExpertsModel  # Replace with your actual import paths

# Load checkpoint
cp = torch.load("../checkpoints/good-5-20/7000.pt", map_location="cpu")
dqn_state = cp["model_state"]

# Initialize models
dqn_model = DQNModel()
experts_model = ExpertsModel(is_train=True)

# Forward pass to initialize LazyLinear with 2304 in_features
dummy_input = torch.randn(1, 4, 1, 128, 128)  # (B, T, C, H, W)
_ = dqn_model(dummy_input)
_ = experts_model(dummy_input, is_ship=torch.tensor([1]))

# Load DQN weights
dqn_model.load_state_dict(dqn_state)

# Copy backbone weights
for dqn_layer, expert_layer in zip(dqn_model.conv, experts_model.backbone):
    if isinstance(dqn_layer, torch.nn.Conv2d):
        expert_layer.weight.data.copy_(dqn_layer.weight.data)
        expert_layer.bias.data.copy_(dqn_layer.bias.data)

# Copy FC weights to both cube_fc and ship_fc
for i, layer in enumerate(dqn_model.fc[1:]):  # Skip Flatten
    if isinstance(layer, torch.nn.Linear):
        for expert_head in [experts_model.cube_fc, experts_model.ship_fc]:
            expert_head[i + 1].weight.data.copy_(layer.weight.data)
            expert_head[i + 1].bias.data.copy_(layer.bias.data)

# Save transferred model
torch.save({"model_state": experts_model.state_dict()}, "../checkpoints/experts_model_init.pt")
print("✅ ExpertsModel weights initialized from DQNModel and saved.")
