import numpy as np
import os

def read_image(img_name):
    base = os.path.basename(img_name)
    identificador, raw = os.path.splitext(base)
    parts = identificador.split('.')

    nom, dades = parts

    dades_separades = dades.split('_')
    tipus,num_components, files, columnes = dades_separades

    if tipus.startswith('ube'):
        signed = "unsigned"
        endian = "big"
        num_bytes = int(tipus[3:])
    elif tipus.startswith('ule'):
        signed = "unsigned"
        endian = "little"
        num_bytes = int(tipus[3:])
    elif tipus.startswith('sbe'):
        signed = "signed"
        endian = "big"
        num_bytes = int(tipus[3:])
    elif tipus.startswith('sle'):
        signed = "signed"
        endian = "little"
        num_bytes = int(tipus[3:])
    else:
        raise ValueError("Tipus desconegut")

    return {
        "files": files,
        "columnes": columnes,
        "num_bytes": num_bytes,
        "signed": signed,
        "endian": endian,
        "nom": nom,
        "components": num_components
    }

diccionario = {}
img = "C:/Users/pablo/PycharmProjects/TCI_project/imatges/03508649.ube16_1_512_512.raw"
diccionario = read_image(img)
def write_copy(img,diccioanrio,augmentar):
    num_bytes = diccionario['num_bytes']
    if augmentar and num_bytes == 1:
        num_bytes = 2
    print(img)
    if diccionario['signed'] == "unsigned" and diccionario['endian'] =="little":
        tipus = "ule"
    elif diccionario['signed'] == "unsigned" and diccionario['endian'] =="big":
        tipus = "ube"
    elif diccionario['signed'] == "signed" and diccionario['endian'] =="little":
        tipus = "sle"
    elif diccionario['signed'] == "signed" and diccionario['endian'] =="big":
        tipus = "sbe"

    num_bits = num_bytes * 8//8
    print(num_bits)
    tipus = tipus + str(num_bits)
    print(tipus)

    nom_copia = diccionario['nom'] + "_copia"
    folder = os.path.dirname(img) or "."
    img_copy = os.path.join(folder,f"{nom_copia}.{tipus}_{diccionario['components']}_{diccionario['files']}_{diccionario['columnes']}.raw")

    print(img_copy)



augmentar = True
write_copy(img,diccionario,augmentar)


#print (diccionario)



