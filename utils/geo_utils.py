import torch

def compute_global_rot_from_local_rot(params, priority, parents, inverse=False):
    '''
    :param params: (B, K, 6)
    :param priority: topk object - values and indices, each is (K,)
    :return:
    '''
    B, K, _ = params.size()

    R = compute_rotation_matrix_from_6d(params)  # (B, K, 3, 3)
    root = priority.indices[0]
    Rglob = dict()
    Rglob[root.item()] = R[:, root]

    for idx in priority.indices:
        if idx == root:
            continue
        else:
            parent = parents[idx] # parent is nearest among ancestors
            if not inverse:
                temp_R = torch.bmm(Rglob[parent.item()], R[:, idx])
            else:
                temp_R = torch.bmm(R[:, idx], Rglob[parent])
            Rglob[idx.item()] = temp_R

    return Rglob  # query : child, value: parent


def normalize_vector( v, return_mag =False):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = v_mag + 1e-10 # to avoid nans
    v_mag = v_mag.reshape(batch,1)
    v = v / v_mag
    if(return_mag):
        return v, v_mag[:,0]
    else:
        return v


# u, v [B, n]
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


def compute_rotation_matrix_from_6d(param):  # parameter range: -inf ~ inf
    # param  : #(B, K, 6)

    *B, C = param.size()
    param = param.reshape(-1, C)

    assert C == 6

    x_raw = param[:, 0:3]  # [B,3]
    y_raw = param[:, 3:6]  # [B,3]

    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # [B, 3, 3]
    matrix = matrix.reshape(*B, 3, 3)  # (B, K, 3, 3)

    return matrix

