import os
import sys
import numpy as np
import json
import napari
from tqdm import tqdm
from tifffile import imread, imwrite
from skimage import exposure, img_as_float 
from napari.utils.colormaps import Colormap # Importación compatible

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
# Esto desactiva el CLAHE solo para este canal (para evitar saturación).
CANAL_PARED_SATURADO = "C01" 

# Definición de vectores de color (RGBA) para la visualización en Napari
COLOR_MAP_VECTORS = {
    "blue": [0, 0, 1, 1],
    "green": [0, 1, 0, 1],
    "red": [1, 0, 0, 1],
    "yellow": [1, 1, 0, 1],
    "magenta": [1, 0, 1, 1],
    "cyan": [0, 1, 1, 1],
    "gray": [1, 1, 1, 1],
}

# Asignación de colores a los nombres de canal de Bio-Formats (C00, C01, etc.)
CHANNEL_COLORS = {
    "C00": "blue",    # Canal 0 (Ej: SR2200)
    "C01": "green",   # Canal 1 (Pared Celular/SYTOX)
    "C02": "red",     # Canal 2 (Ej: EdU)
    "SINGLECHANNEL": "gray",
}

# ====================================================================

def process_tiff_batch(input_dir, output_dir, color_config, normalize, enhance_contrast, save_dtype, show_preview):
    """
    Lee archivos TIFF 3D (separados por canal), aplica procesamiento condicional 
    (Min/Max y CLAHE selectivo), y agrupa para guardar y previsualizar en 3D.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".tif") or f.lower().endswith(".tiff")]
    
    # Agrupar archivos por nombre base
    file_groups = {}
    for f in all_tiff_files:
        # Asume el formato 'nombre_base_C00.tif'
        parts = f.rsplit('_C', 1) 
        if len(parts) > 1:
            base_name = parts[0]
            # Extrae la parte de canal (ej. C00)
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
            
            # ------------------------------------------------------------------
            # CORRECCIÓN DE ERROR: Sanitización de datos (reemplazar NaN/Inf por 0)
            # Esto evita el error 'invalid syntax' en equalize_adapthist.
            data_norm = np.nan_to_num(data_norm)
            # ------------------------------------------------------------------
            
            # LÓGICA CONDICIONAL DE MEJORA DE CONTRASTE (CLAHE)
            should_enhance = enhance_contrast
            
            if original_channel_name == CANAL_PARED_SATURADO:
                should_enhance = False
                print(f"   [INFO] Saltando CLAHE para el canal {CANAL_PARED_SATURADO} (Pared Celular) para evitar saturación.")

            if should_enhance:
                # CLAHE aplicado al stack 3D (slice por slice)
                data_norm = exposure.equalize_adapthist(data_norm)
                
            # Convertir de vuelta al tipo de dato de salida (uint16)
            data_final = (data_norm * np.iinfo(save_dtype).max).astype(save_dtype)
            
            # 2. Guardar el TIFF 3D procesado
            file_name_tiff = f"{base_name}_{original_channel_name}_normalized.tiff"
            save_path_tiff = os.path.join(output_dir, file_name_tiff)
            imwrite(save_path_tiff, data_final)
            
            # 3. Almacenar para Napari
            napari_layers_info.append((data_final, f"{original_channel_name} (Norm)", color))
            
            metadata_list.append({
                "file_name": file_name_tiff,
                "original_channel_id": original_channel_name,
                "colormap": color
            })
            
        # 4. Guardar archivo de metadatos JSON (sin cambios)
        metadata_file_name = f"{base_name}_metadata.json"
        metadata_save_path = os.path.join(output_dir, metadata_file_name)
        with open(metadata_save_path, 'w') as f:
            json.dump(metadata_list, f, indent=4)

        # 5. Previsualización con Napari 3D
        if show_preview and napari_layers_info:
            viewer = napari.Viewer()
            viewer.title = f"QC Final 3D: {base_name}"

            for d, name, color_str in napari_layers_info:
                
                # Creación de Colormap explícito para compatibilidad
                if color_str.lower() in ('gray', 'gris'):
                    custom_colormap = 'gray'
                else:
                    color_vector = COLOR_MAP_VECTORS.get(color_str.lower(), COLOR_MAP_VECTORS['gray'])
                    colors = np.array([[0, 0, 0, 0], color_vector])
                    custom_colormap = Colormap(colors, name=f"{color_str}_direct")

                vmin = d.min()
                vmax = d.max()
                contrast_limits = [vmin, vmax] if vmin != vmax else [0, np.iinfo(save_dtype).max]
                
                # Se agrega como imagen 3D automáticamente
                viewer.add_image(d, 
                                 name=name, 
                                 colormap=custom_colormap, 
                                 contrast_limits=contrast_limits,
                                 blending='additive') 

            viewer.reset_view()
            napari.run()

    print("\n🎯 Procesamiento y Normalización de TIFFs 3D completada.")
    print(f"Archivos finales 3D listos para PlantSeg en: {output_dir_final}")

# -------------------- MAIN --------------------
if __name__ == "__main__":
    
    # 1. Ejecutar el procesamiento
    try:
        process_tiff_batch(
            input_dir_tiff, 
            output_dir_final, 
            CHANNEL_COLORS, 
            normalize, 
            enhance_contrast, 
            save_dtype, 
            show_napari_preview
        )
    except Exception as e:
        print(f"\n❌ Error fatal durante el procesamiento: {e}")