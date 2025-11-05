import os
import sys
import numpy as np
import json
import napari
from tqdm import tqdm
from tifffile import imread, imwrite
from skimage import exposure, img_as_float 

# ====================================================================
# ---------- CONFIGURACIÓN CLAVE (AJUSTAR RUTAS) ----------
# ====================================================================

# Rutas de Archivos (DEBE COINCIDIR CON LA SALIDA DE IMAGEJ)
input_dir_tiff = r"C:\Users\ThinkStation\Desktop\USUARIOS\AaronCastillo\Imagenesprueba\imagenes_tiff_raw"
output_dir_final = r"C:\Users\ThinkStation\Desktop\USUARIOS\AaronCastillo\Imagenesprueba\imagenes_tiff_normalized"

# Opciones de Procesamiento
normalize = True       
enhance_contrast = True # Adaptative Histogram Equalization (CLAHE)
save_dtype = np.uint16 
show_napari_preview = True 

# Definición de colores basada en los nombres de canal de ImageJ (Ch00, Ch01, etc.)
# Esto es para la visualización en Napari.
CHANNEL_COLORS = {
    "CH00": "blue",    # Canal 00 (típicamente DAPI/SR2200)
    "CH01": "green",   # Canal 01 (típicamente SYTOX/Alexa 430)
    "CH02": "red",     # Canal 02 (típicamente EdU/APC)
    "SINGLECHANNEL": "gray",
}

# ====================================================================
# -------------------- FUNCIONES DE PROCESAMIENTO --------------------
# ====================================================================

def process_tiff_batch(input_dir, output_dir, color_config, normalize, enhance_contrast, save_dtype, show_preview):
    """
    Lee archivos TIFF (separados por canal), aplica procesamiento y agrupa 
    para guardar y previsualizar.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".tif") or f.lower().endswith(".tiff")]
    
    # Agrupar archivos por nombre base (antes de _ChXX.tif)
    file_groups = {}
    for f in all_tiff_files:
        # Buscamos la parte del nombre que precede a _ChXX o _C
        parts = f.rsplit('_C', 1) 
        if len(parts) > 1:
            base_name = parts[0]
            channel_info = f.split(base_name)[-1].rsplit('.', 1)[0] # Obtiene _C00, _C01, etc.
        else:
            base_name = os.path.splitext(f)[0]
            channel_info = "" # Canal único
            
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append((f, channel_info))

    print(f"\nDetectados {len(file_groups)} grupos de imágenes para procesar.")

    for base_name, files_list in tqdm(file_groups.items(), desc="Procesando grupos de TIFF"):
        
        napari_layers_info = []
        metadata_list = []
        
        # Procesar cada canal en el grupo
        for file_name, channel_info in sorted(files_list):
            
            original_channel_name = channel_info.upper().replace('_', '') if channel_info else "SINGLECHANNEL"
            
            # Obtener el color del canal usando el nombre simplificado (ej. C00)
            color = color_config.get(original_channel_name, 'gray')
            
            # 1. Leer y Procesar
            file_path = os.path.join(input_dir, file_name)
            data = imread(file_path)
            
            # Convertir a float y aplicar normalización (0-1)
            data_float = img_as_float(data)
            
            if normalize:
                data_norm = (data_float - data_float.min()) / (data_float.max() - data_float.min() + 1e-8)
            else:
                data_norm = data_float
            
            if enhance_contrast:
                # Aplicar CLAHE (equalización adaptativa del histograma)
                data_norm = exposure.equalize_adapthist(data_norm)
                
            # Convertir de vuelta al tipo de dato de salida (uint16)
            data_final = (data_norm * np.iinfo(save_dtype).max).astype(save_dtype)
            
            # 2. Guardar el TIFF procesado
            file_name_tiff = f"{base_name}{channel_info}_normalized.tiff"
            save_path_tiff = os.path.join(output_dir, file_name_tiff)
            imwrite(save_path_tiff, data_final)
            
            # 3. Almacenar para Napari y Metadatos
            napari_layers_info.append((data_final, f"{original_channel_name} (Norm)", color))
            metadata_list.append({
                "file_name": file_name_tiff,
                "original_channel_id": original_channel_name,
                "colormap": color
            })
            
        # Guardar archivo de metadatos JSON
        metadata_file_name = f"{base_name}_metadata.json"
        metadata_save_path = os.path.join(output_dir, metadata_file_name)
        with open(metadata_save_path, 'w') as f:
            json.dump(metadata_list, f, indent=4)

        # 4. Previsualización con Napari (si está activado)
        if show_preview and napari_layers_info:
            viewer = napari.Viewer()
            viewer.title = f"QC Final: {base_name}"

            for d, name, color_str in napari_layers_info:
                
                # Crear Colormap explícito para compatibilidad (usando el vector RGB)
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

    print("\n🎯 Procesamiento y Normalización de TIFFs completada.")
    print(f"Archivos finales en: {output_dir_final}")

# -------------------- MAIN --------------------
if __name__ == "__main__":
    
    # Asegurar que el diccionario COLOR_MAP_VECTORS está disponible (desde el script original)
    # Lo he incluido en este script, pero si usas el código anterior, asegúrate de que esté disponible
    if 'COLOR_MAP_VECTORS' not in globals():
        print("❌ Error: Falta el diccionario COLOR_MAP_VECTORS. Incluye la definición del script anterior.")
        sys.exit(1)

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