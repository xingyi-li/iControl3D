import torch
from third_party.ZoeDepth.zoedepth.utils.misc import pil_to_batched_tensor


def load_zoe_depth(model_type="ZoeD_NK"):
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo = "isl-org/ZoeDepth"
    model_zoe_nk = torch.hub.load(repo, model_type, pretrained=True)
    model = model_zoe_nk.to(device)

    return model


def run_zoe_online(input_img):
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    # # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # repo = "isl-org/ZoeDepth"
    # model_zoe_nk = torch.hub.load(repo, model_type, pretrained=True)
    # model = model_zoe_nk.to(device)
    model = load_zoe_depth(model_type="ZoeD_NK")

    print('=========================using ZoeDepth to compute DPT depth maps...=========================')
    # input
    img = input_img

    X = pil_to_batched_tensor(img).to(device)

    # compute
    with torch.no_grad():
        depth_tensor = model.infer(X)
    print("shape of depth_tensor: ", depth_tensor.shape)
    print("finished")

    return depth_tensor