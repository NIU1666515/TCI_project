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

    arr = np.fromfile(img, dtype=dtype_original)

    return (d,arr)


def write_copy(img, d, augmentar,arr):

    nbytes = d['num_bytes']
    if augmentar:
        nbytes = nbytes * 2

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

def quantitzacio(arr,q):
    arr_flat = arr.tolist()
    val_min = np.min(arr_flat)
    val_max = np.max(arr_flat)

    pas_quant = (val_max - val_min + 1) / q

    arr_quant = [int((val - val_min) // pas_quant) for val in arr_flat]

    return arr_quant

#def descuantitzar(arr_quantitzat,q):



def input_function(img):
    print("Quina acció vols realitzar sobre la imatge?")
    print("1. Llegir")
    print("2. Escriure")
    print("3. Calcular Entropia")
    print("4. Quantitzar")
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
            q = input("Introdueix el valor de quantització:")
            quantitzacio(arr, q)


img = r"C:\Users\pablo\PycharmProjects\TCI_project\imatges\n1_GRAY.ube8_1_2560_2048.raw"


input_function(img)
