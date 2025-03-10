import os
import gdown

os.makedirs('./u2net.pth', exist_ok=True)
os.makedirs('./u2net_portrait.pth', exist_ok=True)

gdown.download('https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ',
    './saved_models/u2net/u2net.pth',
    quiet=False)

gdown.download('https://drive.google.com/uc?id=1IG3HdpcRiDoWNookbncQjeaPN28t90yW',
    './saved_models/u2net_portrait/u2net_portrait.pth',
    quiet=False)
