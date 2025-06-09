#!/bin/bash

# Step 2: Start ssh-agent if not already running
echo "🚀 Starting ssh-agent..."
eval "$(ssh-agent -s)"
sleep 1
# Step 3: Add SSH key to agent
ssh-add ~/.ssh/id_ed25519


echo "✅ Key added to ssh-agent."
sleep 1
# Step 5: Test SSH connection to GitHub
echo -e "\n🧪 Testing SSH connection to GitHub..."
ssh -T git@github.com
