import numpy as np
import os

def read_image(img_name):
    base = os.path.basename(img_name)
    identificador, _ = os.path.splitext(base)
    nom, dades = identificador.split('.', 1)
    tipus, num_components, files, columnes = dades.split('_')
    if tipus.startswith('ube'):
        signed, endian = "unsigned", "big"; bits = int(tipus[3:])
    elif tipus.startswith('ule'):
        signed, endian = "unsigned", "little"; bits = int(tipus[3:])
    elif tipus.startswith('sbe'):
        signed, endian = "signed", "big"; bits = int(tipus[3:])
    elif tipus.startswith('sle'):
        signed, endian = "signed", "little"; bits = int(tipus[3:])
    else:
        raise ValueError(f"Tipus desconegut: {tipus}")
    num_bytes = bits // 8
    return {"files": int(files), "columnes": int(columnes), "num_bytes": num_bytes, "signed": signed, "endian": endian, "nom": nom, "components": int(num_components)}

def write_copy(img, d, augmentar):
    nbytes = d['num_bytes']
    if augmentar and nbytes == 1: nbytes = 2
    if d['signed']=="unsigned" and d['endian']=="little": prefix="ule"
    elif d['signed']=="unsigned" and d['endian']=="big": prefix="ube"
    elif d['signed']=="signed" and d['endian']=="little": prefix="sle"
    else: prefix="sbe"
    bits = nbytes*8
    nom_copia = d['nom']+"_copia"
    folder = os.path.dirname(img) or "."
    img_copy = os.path.join(folder, f"{nom_copia}.{prefix}{bits}_{d['components']}_{d['files']}_{d['columnes']}.raw")
    kind = 'u' if d['signed']=="unsigned" else 'i'
    endian = '>' if d['endian']=="big" else '<'
    arr = np.fromfile(img, dtype=np.dtype(endian+kind+str(d['num_bytes'])))
    if nbytes != d['num_bytes']:
        arr = arr.astype(np.dtype(endian+kind+str(nbytes)))
    arr.tofile(img_copy)
    print("Copia creada:", img_copy)

img = r"C:/Users/pablo/PycharmProjects/TCI_project/imatges/03508649.ube16_1_512_512.raw"
d = read_image(img)
write_copy(img, d, True)
