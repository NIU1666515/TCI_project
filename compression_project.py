import os
import pickle
import numpy as np
import math

def read_dir():
    ruta = "/home/beltix/UNI/4t/TCI/imatges"
    arxius = os.listdir(ruta)
    for i, nom in enumerate(arxius):
        print(i+1,nom)
    print("Quin fitxer vols processar?")
    num_arxiu = int(input("Introdueix el numero de fitxer: ")) - 1
    return os.path.join(ruta, arxius[num_arxiu])

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

    endian_prefix = '>' if d['endian'] == "big" else '<'

    if d['signed'] == "unsigned":
        kind = 'u'
    else:
        kind = 'i'
    if nbytes == 1:
        if d['signed'] == "unsigned":
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.int8)
    else:
        dtype_str = endian_prefix + kind + str(nbytes)
        dtype = np.dtype(dtype_str)
        arr = arr.astype(dtype)
    if d['signed'] == "unsigned" and d['endian'] == "little":
        prefix = "ule"
    elif d['signed'] == "unsigned" and d['endian'] == "big":
        prefix = "ube"
    elif d['signed'] == "signed" and d['endian'] == "little":
        prefix = "sle"
    else:
        prefix = "sbe"

    bits = nbytes * 8
    nom_copia = d['nom'] + "_decoded"
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

def predictor(arr, d):
    H = d["files"] * d["components"]
    W = d["columnes"]
    I = np.asarray(arr, dtype=np.int32).reshape(H, W)
    R = np.empty_like(I, dtype=np.int32)
    R[:, 0] = I[:, 0]
    R[:, 1:] = I[:, 1:] - I[:, :-1]
    R_pos = np.where(R >= 0, R << 1, (-R << 1) - 1)
    return R_pos.ravel()

def reconstruir_predictor(predicted, d):
    H = d["files"] * d["components"]
    W = d["columnes"]
    R_pos = np.asarray(predicted, dtype=np.int32).reshape(H, W)
    R = np.where((R_pos & 1) == 0, R_pos >> 1, -((R_pos + 1) >> 1))
    Y = np.cumsum(R, axis=1)
    return Y.ravel()

# ---------------- Bitstream Arithmetic Coding ----------------
class BitWriter:
    def __init__(self):
        self.buffer = bytearray()
        self.cur_byte = 0
        self.bit_pos = 0

    def write_bit(self, bit):
        self.cur_byte = (self.cur_byte << 1) | (bit & 1)
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.buffer.append(self.cur_byte)
            self.cur_byte = 0
            self.bit_pos = 0

    def flush(self):
        if self.bit_pos > 0:
            self.cur_byte <<= (8 - self.bit_pos)
            self.buffer.append(self.cur_byte)
            self.cur_byte = 0
            self.bit_pos = 0


class BitReader:
    def __init__(self, buf):
        self.buffer = buf
        self.byte_pos = 0
        self.bit_pos = 0

    def read_bit(self):
        if self.byte_pos >= len(self.buffer):
            return 0
        bit = (self.buffer[self.byte_pos] >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos == 8:
            self.bit_pos = 0
            self.byte_pos += 1
        return bit


class ArithmeticCoder:
    def __init__(self):
        self.low = 0
        self.high = 0xFFFFFFFF
        self.underflow = 0

    def encode_symbol(self, symbol, cum_freq, bw):
        range_ = self.high - self.low + 1
        self.high = self.low + (range_ * cum_freq[symbol + 1]) // cum_freq[-1] - 1
        self.low  = self.low + (range_ * cum_freq[symbol]) // cum_freq[-1]

        while True:
            if (self.high & 0x80000000) == (self.low & 0x80000000):
                bit = (self.high >> 31) & 1
                bw.write_bit(bit)
                for _ in range(self.underflow):
                    bw.write_bit(1 - bit)
                self.underflow = 0
                self.low = (self.low << 1) & 0xFFFFFFFF
                self.high = ((self.high << 1) & 0xFFFFFFFF) | 1
            elif (self.low & 0x40000000) and not (self.high & 0x40000000):
                self.underflow += 1
                self.low &= 0x3FFFFFFF
                self.high |= 0x40000000
                self.low = (self.low << 1) & 0xFFFFFFFF
                self.high = ((self.high << 1) & 0xFFFFFFFF) | 1
            else:
                break

    def finish(self, bw):
        self.underflow += 1
        if self.low < 0x40000000:
            bw.write_bit(0)
            for _ in range(self.underflow):
                bw.write_bit(1)
        else:
            bw.write_bit(1)
            for _ in range(self.underflow):
                bw.write_bit(0)
        bw.flush()


class ArithmeticDecoder:
    def __init__(self, br):
        self.low = 0
        self.high = 0xFFFFFFFF
        self.code = 0
        for _ in range(32):
            self.code = (self.code << 1) | br.read_bit()
        self.br = br

    def decode_symbol(self, cum_freq):
        range_ = self.high - self.low + 1
        value = ((self.code - self.low + 1) * cum_freq[-1] - 1) // range_

        left, right = 0, len(cum_freq) - 1
        while left < right - 1:
            mid = (left + right) // 2
            if cum_freq[mid] <= value:
                left = mid
            else:
                right = mid
        symbol = left

        self.high = self.low + (range_ * cum_freq[symbol + 1]) // cum_freq[-1] - 1
        self.low  = self.low + (range_ * cum_freq[symbol]) // cum_freq[-1]

        while True:
            if (self.high & 0x80000000) == (self.low & 0x80000000):
                self.low = (self.low << 1) & 0xFFFFFFFF
                self.high = ((self.high << 1) & 0xFFFFFFFF) | 1
                self.code = ((self.code << 1) & 0xFFFFFFFF) | self.br.read_bit()
            elif (self.low & 0x40000000) and not (self.high & 0x40000000):
                self.low &= 0x3FFFFFFF
                self.high |= 0x40000000
                self.low = (self.low << 1) & 0xFFFFFFFF
                self.high = ((self.high << 1) & 0xFFFFFFFF) | 1
                self.code = (((self.code ^ 0x40000000) << 1) & 0xFFFFFFFF) | self.br.read_bit()
            else:
                break

        return symbol


def compute_cum_freq(data):
    max_symbol = int(max(data)) + 1
    freq = [0] * max_symbol
    for s in data:
        freq[s] += 1
    cum = [0] * (len(freq) + 1)
    for i in range(len(freq)):
        cum[i + 1] = cum[i] + freq[i]
    return cum

# ---------------- Codificador / Decodificador aritmètic sobre arrays ----------------
def codificador_aritmetic(arr, d, img_path,q):
    arr = np.asarray(arr, dtype=np.int32)
    N = len(arr)
    cum_freq = compute_cum_freq(arr)
    bw = BitWriter()
    coder = ArithmeticCoder()
    for s in arr:
        coder.encode_symbol(int(s), cum_freq, bw)
    coder.finish(bw)

    bitstream = bw.buffer

    paquet = {
        "cum_freq": cum_freq,
        "N": N,
        "bitstream": bytes(bitstream),
        "header": d,   # 👈 aquí guardem la capçalera
        "q": int(q),
    }

    folder = os.path.dirname(img_path) or "."
    base = os.path.basename(img_path)
    nombre_sin_ext, _ = os.path.splitext(base)
    tci_path = os.path.join(folder, nombre_sin_ext + ".tci")

    with open(tci_path, "wb") as f:
        pickle.dump(paquet, f)

    print("Fitxer .tci creat:", tci_path)
    return tci_path

def decodificador_aritmetic(tci_path):
    folder = os.path.dirname(tci_path) or "."
    base = os.path.basename(tci_path)
    nom_sense_ext, _ = os.path.splitext(base)
    tci_path = os.path.join(folder, nom_sense_ext + ".tci")

    with open(tci_path, "rb") as f:
        paquet = pickle.load(f)

    cum_freq = paquet["cum_freq"]
    N = paquet["N"]
    bitstream = paquet["bitstream"]
    d = paquet["header"]  
    q = paquet["q"]

    br = BitReader(bytearray(bitstream))
    decoder = ArithmeticDecoder(br)

    data = [decoder.decode_symbol(cum_freq) for _ in range(N)]
    data = np.array(data, dtype=np.int32)
    return data,d,q

def main_process():
    print("Quina acció vols realitzar sobre la imatge?")
    print("1. Comprimir")
    print("2. Descomprimir")
    option = input("Introdueix l'acció:")
    match option:
        case "1":
            img=read_dir()
            d,arr = read_image(img)
            print("Entropia arxiu:")
            entropia_0(arr)
            q = input("Introdueix el valor de quantització: ")
            arr_quantitzat = quantitzacio(arr, q)
            arr_predict=predictor(arr_quantitzat,d)
            codificador_aritmetic(arr_predict,d,img,q)

        case "2":
            tci_img=read_dir()
            arr_decodificat,d,q=decodificador_aritmetic(tci_img)
            arr_despredict=reconstruir_predictor(arr_decodificat,d)
            arr_desquantitzat=desquantitzacio(arr_despredict,q)
            print("Entropia arxiu processat:")
            entropia_0(arr_desquantitzat)
            write_copy(tci_img, d, False, arr_desquantitzat)


main_process()