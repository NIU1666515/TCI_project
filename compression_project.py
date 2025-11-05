import os
import numpy as np
import math

def read_image(img_name):
    base = os.path.basename(img_name)
    identificador, _ = os.path.splitext(base)
    nom, dades = identificador.split('.', 1)
    tipus, num_components, files, columnes = dades.split('_')

    if tipus.startswith('ube'):
        signed, endian = "unsigned", "big"
        bits = int(tipus[3:])
    elif tipus.startswith('ule'):
        signed, endian = "unsigned", "little"
        bits = int(tipus[3:])
    elif tipus.startswith('sbe'):
        signed, endian = "signed", "big"
        bits = int(tipus[3:])
    elif tipus.startswith('sle'):
        signed, endian = "signed", "little"
        bits = int(tipus[3:])
    else:
        raise ValueError(f"Tipus desconegut: {tipus}")

    num_bytes = bits // 8

    d = {
        "files": int(files),
        "columnes": int(columnes),
        "num_bytes": num_bytes,
        "signed": signed,
        "endian": endian,
        "nom": nom,
        "components": int(num_components)
    }

    kind = 'u' if d['signed'] == "unsigned" else 'i'
    endian = '>' if d['endian'] == "big" else '<'
    dtype_original = np.dtype(endian + kind + str(d['num_bytes']))

    arr = np.fromfile(img_name, dtype=dtype_original)

    return (d,arr)


def write_copy(img, d, augmentar, arr):
    nbytes = d['num_bytes']
    if augmentar:
        nbytes = nbytes * 2

    if d['signed'] == "unsigned":
        if nbytes == 1:
            arr = arr.astype(np.uint8)
        elif nbytes == 2:
            arr = arr.astype(np.uint16)
        else:
            arr = arr.astype(np.uint32)
    else:
        if nbytes == 1:
            arr = arr.astype(np.int8)
        elif nbytes == 2:
            arr = arr.astype(np.int16)
        else:
            arr = arr.astype(np.int32)

    if d['signed'] == "unsigned" and d['endian'] == "little":
        prefix = "ule"
    elif d['signed'] == "unsigned" and d['endian'] == "big":
        prefix = "ube"
    elif d['signed'] == "signed" and d['endian'] == "little":
        prefix = "sle"
    else:
        prefix = "sbe"

    bits = nbytes * 8
    nom_copia = d['nom'] + "_copia"
    folder = os.path.dirname(img) or "."

    img_copy = os.path.join(
        folder,
        f"{nom_copia}.{prefix}{bits}_{d['components']}_{d['files']}_{d['columnes']}.raw"
    )

    arr.tofile(img_copy)
    print("Copia creada:", img_copy)


def probabilitats(arr):
    arr_flat = arr.tolist()
    total = len(arr_flat)

    p_marg = {}
    p_conj = {}
    for i in range(total):
        p_marg[arr_flat[i]] = p_marg.get(arr_flat[i], 0) + 1
        if i < total - 1:
            pair = (arr_flat[i], arr_flat[i + 1])
            p_conj[pair] = p_conj.get(pair, 0) + 1

    for k in p_marg:
        p_marg[k] /= total
    for k in p_conj:
        p_conj[k] /= (total - 1)

    p_cond = {(b, a): p_conj[(a, b)] / p_marg[a] for (a, b) in p_conj}

    return p_marg, p_conj, p_cond


def entropia_0(arr):
    p_marg, _, _ = probabilitats(arr)

    entropy = -sum(prob * math.log2(prob) for prob in p_marg.values())

    print("Entropia de la imatge:" , entropy , " bits/píxel")
    return entropy

def entropia_1(arr):
    _, p_conjunta, p_condicional = probabilitats(arr)
    entropy = 0.0
    for (pixel_prev, pixel_curr), p_conj_val in p_conjunta.items():
        p_cond_val = p_condicional[(pixel_curr, pixel_prev)]
        entropy -= p_conj_val * math.log2(p_cond_val)
    print("Entropia condicional respecte el pixel anterior:", entropy, "bits/píxel")
    return entropy

def quantitzacio(arr, q):
    q = int(q)
    arr = np.asarray(arr, dtype=int)
    arr_q = np.round(arr / q).astype(int)
    return arr_q

def desquantitzacio(arr_q, q):
    q = int(q)
    arr_q = np.asarray(arr_q, dtype=int)
    arr_rec = np.round(arr_q * q).astype(int)
    return arr_rec

def calcul_pae(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return int(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64))))

def calcul_mse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean((a - b) ** 2))

def calcul_psnr(a, b, max_val=255):
    m = calcul_mse(a, b)
    if m == 0:
        return float('mse_false')
    return 10 * math.log10((max_val * max_val) / m)

def predictor_izquierda(arr, d):
    H, W = d["files"], d["columnes"]
    I = np.asarray(arr, dtype=np.int32).reshape(H, W)
    R = np.empty_like(I, dtype=np.int32)
    R[:, 0] = I[:, 0]
    R[:, 1:] = I[:, 1:] - I[:, :-1]

    return R.ravel()

def reconstruir_izquierda(residual, d):
    H, W = d["files"], d["columnes"]
    R = np.asarray(residual, dtype=np.int32).reshape(H, W)
    Y = np.cumsum(R, axis=1)
    return Y.ravel()


def input_function(img):
    print("Quina acció vols realitzar sobre la imatge?")
    print("1. Llegir")
    print("2. Escriure")
    print("3. Calcular Entropia")
    print("4. Quantitzar")
    print("5. Quantitzar i Desquantizar")
    print("6. Calculs")
    print("7. Predictor")
    option = input("Introdueix l'acció:")

    match option:
        case "1":
            d,arr = read_image(img)
        case "2":
            d, arr = read_image(img)
            augmentar = False

            if d['num_bytes'] == 1:
                resposta = input("Vols augmentar a 2 bytes? (Y/N): ").strip().upper()
                augmentar = (resposta == "Y")

            write_copy(img, d, augmentar, arr)
        case "3":
            print("1. Entropia 0")
            print("2. Entropia 1")
            entropy = input("Introdueix la entropia: ")
            d, arr = read_image(img)
            if(entropy == "1"):
                entropia_0(arr)
            else: entropia_1(arr)
        case "4":
            d, arr = read_image(img)
            q = input("Introdueix el valor de quantització: ")
            arr_quantitzat = quantitzacio(arr, q)
            write_copy(img, d, False, arr_quantitzat)
            entropia_0(arr_quantitzat)
        case "5":
            d, arr = read_image(img)
            q = input("Introdueix el valor de quantització: ")
            arr_quantitzat = quantitzacio(arr, q)
            arr_desquantitzat = desquantitzacio(arr_quantitzat, q)
            write_copy(img, d, False, arr_desquantitzat)
            entropia_0(arr_desquantitzat)
        case "6":
            d, arr = read_image(img)
            d4, arr_q = read_image(img_copia)
            print("1. PAE")
            print("2. MSE")
            print("3. PSNR")
            option_calcul = input("Introdueix el calcul:")
            match option_calcul:
                case "1": 
                    pae_res = calcul_pae(arr, arr_q)
                    print("Calcul del PAE: ", pae_res)
                case "2":
                    mse_res = calcul_mse(arr, arr_q)
                    print("Calcul del MSE", mse_res)
                case "3":
                    psnr_res = calcul_psnr(arr, arr_q)
                    print("Calcul del PSNR", psnr_res)
        case "7":
            d, arr = read_image(img)
            arr_predict=predictor_izquierda(arr,d)
            q = input("Introdueix el valor de quantització: ")
            arr_quantitzat = quantitzacio(arr_predict, q)
            arr_desquantitzat = desquantitzacio(arr_quantitzat, q)
            arr_despredict=reconstruir_izquierda(arr_desquantitzat,d)
            write_copy(img, d, False, arr_despredict)

img_= r"/home/beltix/UNI/4t/TCI/imatges/n1_GRAY_copia.ube8_1_2560_2048.raw"

img = r"/home/beltix/UNI/4t/TCI/imatges/n1_GRAY.ube8_1_2560_2048.raw"
img_copia = r"/home/beltix/UNI/4t/TCI/imatges/n1_GRAY_copia.ube8_1_2560_2048.raw"

input_function(img)
