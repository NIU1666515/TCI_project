import os
import numpy as np


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

def entropy_calculator(arr):
    """Calcula la entropía del array de imagen."""
    # Aplanar el array (por si es multidimensional)
    arr_flat = arr.flatten()

    # Calcular histograma con todos los posibles valores (0-255 o según el tipo)
    hist, _ = np.histogram(arr_flat, bins=256, range=(0, 255))

    # Normalizar para obtener probabilidades
    p = hist / np.sum(hist)

    # Evitar log(0)
    p = p[p > 0]

    # Calcular entropía
    entropy = -np.sum(p * np.log2(p))

    print(f"Entropia de la imatge: {entropy:.4f} bits/píxel")
    return entropy


def input_function(img):
    print("Quina acció vols realitzar sobre la imatge?")
    print("1. Llegir")
    print("2. Escriure")
    print("3. Calcular Entropia")
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
            d, arr = read_image(img)
            entropy_calculator(arr)


img = r"C:\Users\pablo\PycharmProjects\TCI_project\imatges\n1_GRAY.ube8_1_2560_2048.raw"


input_function(img)
