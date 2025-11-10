import os
import sys
import numpy as np
import json
import napari
from tqdm import tqdm
from tifffile import imread, imwrite
from skimage import exposure, img_as_float 
from napari.utils.colormaps import Colormap

# ====================================================================
# ---------- CONFIGURACIÓN CLAVE (AJUSTAR RUTAS Y CANALES) ----------
# ====================================================================

# Rutas de Archivos (DEBEN COINCIDIR CON LA SALIDA DE FIJI)
input_dir_tiff = r"C:\Users\ThinkStation\Desktop\USUARIOS\AaronCastillo\Imagenesprueba\imagenes_tiff_raw"
output_dir_final = r"C:\Users\ThinkStation\Desktop\USUARIOS\AaronCastillo\Imagenesprueba\imagenes_tiff_normalized"

# Ajustes de Procesamiento
normalize = True       
enhance_contrast = True # Aplicar CLAHE por defecto
save_dtype = np.uint16 
show_napari_preview = True 

# !!! IDENTIFICADOR DEL CANAL DE PARED CELULAR SATURADO !!!
# El Canal C01 NO tendrá CLAHE aplicado.
CANAL_PARED_SATURADO = "C01" 

# Definición de vectores de color (RGBA) para la visualización en Napari
COLOR_MAP_VECTORS = {
    "blue": [0, 0, 1, 1],
    "green": [0, 1, 0, 1],
    "red": [1, 0, 0, 1],
    "gray": [1, 1, 1, 1],
}

# Asignación de colores a los nombres de canal de Bio-Formats
CHANNEL_COLORS = {
    "C00": "blue",    
    "C01": "green",   # <-- Pared Celular (Color de visualización)
    "C02": "red",     
    "SINGLECHANNEL": "gray",
}

# ====================================================================

def process_tiff_batch(input_dir, output_dir, color_config, normalize, enhance_contrast, save_dtype, show_preview):
    """
    Lee archivos TIFF 3D separados por canal, aplica normalización Min/Max, 
    CLAHE selectivo (excluyendo el canal saturado), y guarda el resultado.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".tif") or f.lower().endswith(".tiff")]
    
    # Lógica de agrupamiento de archivos (agrupando C00, C01, C02 por nombre base)
    file_groups = {}
    for f in all_tiff_files:
        parts = f.rsplit('_C', 1) 
        if len(parts) > 1:
            base_name = parts[0]
            channel_info = f.split(base_name)[-1].split('.', 1)[0].split('_')[-1].upper()
        else:
            base_name = os.path.splitext(f)[0]
            channel_info = "SINGLECHANNEL"
            
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append((f, channel_info))

    print(f"\nDetectados {len(file_groups)} grupos de imágenes 3D para procesar.")

    for base_name, files_list in tqdm(file_groups.items(), desc="Procesando grupos de TIFF 3D"):
        
        napari_layers_info = []
        metadata_list = []
        
        # Procesar cada canal en el grupo
        for file_name, original_channel_name in sorted(files_list):
            
            color = color_config.get(original_channel_name, 'gray')
            file_path = os.path.join(input_dir, file_name)
            data = imread(file_path) # Leer el Stack 3D completo
            
            # Convertir a float (0-1)
            data_float = img_as_float(data)
            
            # Normalización Min/Max
            if normalize:
                min_val = data_float.min()
                max_val = data_float.max()
                data_norm = (data_float - min_val) / (max_val - min_val + 1e-8)
            else:
                data_norm = data_float
            
            # --- CORRECCIÓN CRÍTICA ---
            # Sanitización de datos (reemplazar NaN/Inf por 0) para evitar errores de CLAHE.
            data_norm = np.nan_to_num(data_norm)
            
            # LÓGICA CONDICIONAL DE MEJORA DE CONTRASTE (CLAHE)
            should_enhance = enhance_contrast
            
            # Desactivar CLAHE si es el canal de pared saturado
            if original_channel_name == CANAL_PARED_SATURADO:
                should_enhance = False
                print(f"   [INFO] Saltando CLAHE para el canal {CANAL_PARED_SATURADO} (Pared Celular) para evitar saturación.")

            if should_enhance:
                # CLAHE aplicado al stack 3D (slice por slice)
                data_norm = exposure.equalize_adapthist(data_norm)
                
            # Convertir de vuelta al tipo de dato de salida (uint16)
            data_final = (data_norm * np.iinfo(save_dtype).max).astype(save_dtype)
            
            # Guardar el TIFF 3D procesado
            file_name_tiff = f"{base_name}_{original_channel_name}_normalized.tiff"
            save_path_tiff = os.path.join(output_dir, file_name_tiff)
            imwrite(save_path_tiff, data_final)
            
            # Almacenar para Napari
            napari_layers_info.append((data_final, f"{original_channel_name} (Norm)", color))
            
            metadata_list.append({
                "file_name": file_name_tiff,
                "original_channel_id": original_channel_name,
                "colormap": color
            })
            
        # Previsualización con Napari 3D (Control de Calidad)
        if show_preview and napari_layers_info:
            viewer = napari.Viewer()
            viewer.title = f"QC Final 3D: {base_name}"

            for d, name, color_str in napari_layers_info:
                
                if color_str.lower() in ('gray', 'gris'):
                    custom_colormap = 'gray'
                else:
                    color_vector = COLOR_MAP_VECTORS.get(color_str.lower(), COLOR_MAP_VECTORS['gray'])
                    colors = np.array([[0, 0, 0, 0], color_vector])
                    custom_colormap = Colormap(colors, name=f"{color_str}_direct")

                vmin = d.min()
                vmax = d.max()
                contrast_limits = [vmin, vmax] if vmin != vmax else [0, np.iinfo(save_dtype).max]
                
                viewer.add_image(d, 
                                 name=name, 
                                 colormap=custom_colormap, 
                                 contrast_limits=contrast_limits,
                                 blending='additive') 

            viewer.reset_view()
            napari.run()

    print("\n🎯 Procesamiento y Normalización de TIFFs 3D completada.")
