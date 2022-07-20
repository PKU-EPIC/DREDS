from mathutils import Vector
import bpy
import random
import math

def modify_material(mat_links, mat_nodes, material_name, is_texture=False, orign_base_color=None, tex_node=None):
    ## Random: mix ratio
    tex_mix_prop = random.uniform(0.7, 0.95)
    mix_prop = random.uniform(0.6, 0.9)
    ## Random: for one type of material, keep/Not keep the texture or color of the objects itself = 3:1
    transfer_rand = random.randint(0,3)
    if transfer_rand == 0:
        transfer_flag = False
    else:
        transfer_flag = True
    ## Random: Principled BSDF
    bsdfnode_list = [n for n in mat_nodes if isinstance(n, bpy.types.ShaderNodeBsdfPrincipled)]
    if bsdfnode_list != []:
        for bsdfnode in bsdfnode_list:
            if not bsdfnode.inputs[4].links:    # metallic
                src_value = bsdfnode.inputs[4].default_value
                if material_name.split("_")[0] == "metal":
                    new_value = src_value + random.uniform(-0.05, 0.05)
                elif material_name.split("_")[0] == "porcelain":
                    new_value = src_value + random.uniform(-0.05, 0.1)
                elif material_name.split("_")[0] == "plasticsp":
                    new_value = src_value + random.uniform(-0.05, 0.1)
                else:
                    new_value = src_value + random.uniform(-0.05, 0.05)
                if new_value > 1.0: new_value = 1.0
                elif new_value < 0: new_value = 0.0
                bsdfnode.inputs[4].default_value = new_value 
            if not bsdfnode.inputs[5].links:    # specular
                src_value = bsdfnode.inputs[5].default_value
                #if material_name.split("_")[0] == "metal":
                new_value = src_value + random.uniform(0, 0.3)
                if new_value > 1.0: new_value = 1.0
                elif new_value < 0: new_value = 0.0
                bsdfnode.inputs[5].default_value = new_value
            if not bsdfnode.inputs[6].links:    # specularTint
                src_value = bsdfnode.inputs[6].default_value
                new_value = src_value + random.uniform(-1, 1)
                if new_value > 1.0: new_value = 1.0
                elif new_value < 0: new_value = 0.0
                bsdfnode.inputs[6].default_value = new_value
            if not bsdfnode.inputs[7].links:    # roughness
                src_value = bsdfnode.inputs[7].default_value
                if material_name.split("_")[0] == "metal" or material_name.split("_")[0] == "porcelain" or material_name.split("_")[0] == "plasticsp" or material_name.split("_")[0] == "paintsp":
                    new_value = src_value + random.uniform(-0.1, 0.03)
                else:
                    new_value = src_value + random.uniform(-0.03, 0.1)
                if new_value > 1.0: new_value = 1.0
                elif new_value < 0: new_value = 0.0
                bsdfnode.inputs[7].default_value = new_value
            if not bsdfnode.inputs[8].links:    # anisotropic
                src_value = bsdfnode.inputs[8].default_value
                new_value = src_value + random.uniform(-0.1, 0.1)
                if new_value > 1.0: new_value = 1.0
                elif new_value < 0: new_value = 0.0
                bsdfnode.inputs[8].default_value = new_value
            if not bsdfnode.inputs[9].links:    # anisotropicRotation
                src_value = bsdfnode.inputs[9].default_value
                new_value = src_value + random.uniform(-0.3, 0.3)
                if new_value > 1.0: new_value = 1.0
                elif new_value < 0: new_value = 0.0
                bsdfnode.inputs[9].default_value = new_value
            if not bsdfnode.inputs[10].links:    # sheen
                src_value = bsdfnode.inputs[10].default_value
                new_value = src_value + random.uniform(-0.1, 0.1)
                if new_value > 1.0: new_value = 1.0
                elif new_value < 0: new_value = 0.0
                bsdfnode.inputs[10].default_value = new_value
            if not bsdfnode.inputs[11].links:    # sheenTint
                src_value = bsdfnode.inputs[11].default_value
                new_value = src_value + random.uniform(-0.2, 0.2)
                if new_value > 1.0: new_value = 1.0
                elif new_value < 0: new_value = 0.0
                bsdfnode.inputs[11].default_value = new_value
            if not bsdfnode.inputs[12].links:    # clearcoat
                src_value = bsdfnode.inputs[12].default_value
                new_value = src_value + random.uniform(-0.2, 0.2)
                if new_value > 1.0: new_value = 1.0
                elif new_value < 0: new_value = 0.0
                bsdfnode.inputs[12].default_value = new_value
            if not bsdfnode.inputs[13].links:    # clearcoatGloss
                src_value = bsdfnode.inputs[13].default_value
                new_value = src_value + random.uniform(-0.2, 0.2)
                if new_value > 1.0: new_value = 1.0
                elif new_value < 0: new_value = 0.0
                bsdfnode.inputs[13].default_value = new_value

    ## metal
    if material_name == "metal_0":
        ## Random Parameters
        # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.95, 1.00)       # metallic
        # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.3, 1.0)         # specular
        # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
        mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 1.0)         # anisotropic
        # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 1.0)         # anisotropicRotation
        # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 1.0)         # clearcoat
        # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.3, 1.0)         # clearcoatGloss

        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

            mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
            mat_links.new(mat_nodes["Image Texture.002"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
    elif material_name == "metal_1":
        ## Random Parameters
        # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.9, 1.00)        # metallic
        # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
        # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.5, 1.0)         # specularTint
        mat_nodes["Principled BSDF"].inputs[7].default_value = random.uniform(0.08, 0.25)         # roughness
        mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.04, 0.5)         # anisotropic
        # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.3, 0.7)         # anisotropicRotation
        # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.8, 1.0)         # clearcoat
        # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 1.0)         # clearcoatGloss
        
        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

            mat_links.new(mat_nodes["Tangent"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[22]) 

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
    elif material_name == "metal_10":
        ## Random Parameters
        # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
        # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
        mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 0.5)         # anisotropic
        # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.3, 0.7)         # anisotropicRotation
        # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 1.0)         # clearcoat
        # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 1.0)         # clearcoatGloss
        
        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            bsdf_new.location = Vector((-800, 0))
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'
            mix_new.location = Vector((-800, 0))

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7

            mat_links.new(mat_nodes["Image Texture"].outputs[1], mat_nodes["Principled BSDF-new"].inputs[19])
            mat_links.new(mat_nodes["Image Texture.001"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[4])
            mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
            mat_links.new(mat_nodes["ColorRamp"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
    elif material_name == "metal_11":
        ## Random Parameters
        # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
        # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
        mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 0.8)         # anisotropic
        # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 0.8)         # anisotropicRotation
        # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 1.0)         # clearcoat
        # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 1.0)         # clearcoatGloss
        
        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

            mat_links.new(mat_nodes["Image Texture.001"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[4])    
            mat_links.new(mat_nodes["Image Texture.002"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])  
            mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])                  
    elif material_name == "metal_12":
        ## Random Parameters
        # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.95, 1.00)       # metallic
        # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
        # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
        mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 0.8)         # anisotropic
        # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 0.8)         # anisotropicRotation
        # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 1.0)         # clearcoat
        # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 1.0)         # clearcoatGloss
        
        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

            mat_links.new(mat_nodes["ColorRamp"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])    
            mat_links.new(mat_nodes["Reroute.006"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])  

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
    elif material_name == "metal_13":
        ## Random Parameters
        # mat_nodes["Principled BSDF.001"].inputs[4].default_value = random.uniform(0.95, 1.00)       # metallic
        # mat_nodes["Principled BSDF.001"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
        # mat_nodes["Principled BSDF.001"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
        mat_nodes["Principled BSDF.001"].inputs[8].default_value = random.uniform(0.3, 0.7)         # anisotropic
        # mat_nodes["Principled BSDF.001"].inputs[9].default_value = random.uniform(0.0, 0.8)         # anisotropicRotation
        # mat_nodes["Principled BSDF.001"].inputs[12].default_value = random.uniform(0.0, 1.0)         # clearcoat
        # mat_nodes["Principled BSDF.001"].inputs[13].default_value = random.uniform(0.0, 1.0)         # clearcoatGloss

        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF.001"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

            mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20]) 
            mat_links.new(mat_nodes["Mix.001"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])  

            mat_links.new(mat_nodes["Principled BSDF.001"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output.001"].inputs["Surface"])  
    elif material_name == "metal_14":
        ## Random Parameters
        # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.95, 1.00)       # metallic
        # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
        # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
        mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 0.5)         # anisotropic
        # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 0.5)         # anisotropicRotation
        # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 1.0)         # clearcoat
        # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 1.0)         # clearcoatGloss
        
        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.85 
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7 
            
            mat_links.new(mat_nodes["Group"].outputs[1], mat_nodes["Principled BSDF-new"].inputs[7])  
            mat_links.new(mat_nodes["Group"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])   

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
    elif material_name == "metal_2":
        ## Random Parameters
        # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.95, 1.00)       # metallic
        # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
        # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.5, 1.0)         # specularTint
        mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 0.95)        # anisotropic
        # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 1.0)         # anisotropicRotation
        # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 1.0)        # clearcoat
        # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 1.0)        # clearcoatGloss

        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

            mat_links.new(mat_nodes["Image Texture.003"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7]) 
            mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])  

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])       
    elif material_name == "metal_3":
        ## Random Parameters
        # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.95, 1.00)       # metallic
        # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
        # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.5, 1.0)         # specularTint
        mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 0.2)        # anisotropic
        # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 1.0)         # anisotropicRotation
        # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 1.0)        # clearcoat
        # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 1.0)        # clearcoatGloss
        mat_nodes["Gamma"].inputs[1].default_value = random.uniform(3.0, 4.0)

        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

            mat_links.new(mat_nodes["Gamma"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7]) 
            mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])  

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])       
    elif material_name == "metal_4":
        ## Random Parameters
        # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.95, 1.00)       # metallic
        # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.1, 0.5)         # specular
        # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
        mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.0, 0.2)        # anisotropic
        # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 1.0)         # anisotropicRotation
        # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 0.5)        # clearcoat
        # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 0.5)        # clearcoatGloss
 
        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

            mat_links.new(mat_nodes["ColorRamp"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7]) 
            mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])  

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])  
    elif material_name == "metal_5":
        ## Random Parameters
        # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.98, 1.00)       # metallic
        # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.2, 0.4)         # specular
        # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
        mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.6, 0.9)        # anisotropic
        # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 1.0)         # anisotropicRotation
        # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.8, 1.0)        # clearcoat
        # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 0.3)        # clearcoatGloss

        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7  

            mat_links.new(mat_nodes["Voronoi Texture"].outputs[1], mat_nodes["Principled BSDF-new"].inputs[7]) 
            mat_links.new(mat_nodes["Tangent"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[22])   
            mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[21])  

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])  
    elif material_name == "metal_6":
        ## Random Parameters
        # mat_nodes["BSDF guidé"].inputs[4].default_value = random.uniform(0.98, 1.00)       # metallic
        # mat_nodes["BSDF guidé"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
        # mat_nodes["BSDF guidé"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
        mat_nodes["BSDF guidé"].inputs[8].default_value = random.uniform(0.0, 0.2)        # anisotropic
        # mat_nodes["BSDF guidé"].inputs[9].default_value = random.uniform(0.0, 1.0)         # anisotropicRotation
        # mat_nodes["BSDF guidé"].inputs[12].default_value = random.uniform(0.0, 0.3)        # clearcoat
        # mat_nodes["BSDF guidé"].inputs[13].default_value = random.uniform(0.0, 0.3)        # clearcoatGloss
        mat_nodes["Valeur"].outputs[0].default_value = random.uniform(0.1, 0.3)

        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            bsdf_new.location = Vector((-800, 0))
            for key, input in enumerate(mat_nodes["BSDF guidé"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'
            mix_new.location = Vector((-800, 0))

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9  
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7 

            mat_links.new(mat_nodes["Mélanger.002"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

            mat_links.new(mat_nodes["BSDF guidé"].outputs[0], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs[0], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Sortie de matériau"].inputs[0])
    elif material_name == "metal_7":
        ## Random Parameters
        # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.98, 1.00)       # metallic
        # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
        # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
        mat_nodes["Principled BSDF"].inputs[8].default_value = random.uniform(0.7, 0.9)        # anisotropic
        # mat_nodes["Principled BSDF"].inputs[9].default_value = random.uniform(0.0, 1.0)         # anisotropicRotation
        # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 0.3)        # clearcoat
        # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 0.3)        # clearcoatGloss
        
        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            #bsdf_new.location = Vector((-800, 0))
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'
            #mix_new.location = Vector((-800, 0))

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9 
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.7

            mat_links.new(mat_nodes["Reroute.001"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])
            mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
            mat_links.new(mat_nodes["Tangent"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[22])

            mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
    elif material_name == "metal_8":
        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value
            bsdf_1_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_1_new.name = 'Principled BSDF-1-new'
            for key, input in enumerate(mat_nodes["Principled BSDF.001"].inputs):
                bsdf_1_new.inputs[key].default_value = input.default_value
            bsdf_2_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_2_new.name = 'Principled BSDF-2-new'
            for key, input in enumerate(mat_nodes["Principled BSDF.002"].inputs):
                bsdf_2_new.inputs[key].default_value = input.default_value
            bsdf_3_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_3_new.name = 'Principled BSDF-3-new'
            for key, input in enumerate(mat_nodes["Principled BSDF.003"].inputs):
                bsdf_3_new.inputs[key].default_value = input.default_value
            
            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'
            mix_1_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_1_new.name = 'Mix Shader-1-new'
            mix_2_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_2_new.name = 'Mix Shader-2-new'
            mix_3_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_3_new.name = 'Mix Shader-3-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs[0])
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-1-new"].inputs[0])
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-2-new"].inputs[0])
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-3-new"].inputs[0])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.6
                mat_nodes["Mix Shader-1-new"].inputs[0].default_value = 0.6
                mat_nodes["Mix Shader-2-new"].inputs[0].default_value = 0.6
                mat_nodes["Mix Shader-3-new"].inputs[0].default_value = 0.6
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Principled BSDF-1-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Principled BSDF-2-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Principled BSDF-3-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.5
                mat_nodes["Mix Shader-1-new"].inputs[0].default_value = 0.5
                mat_nodes["Mix Shader-2-new"].inputs[0].default_value = 0.5
                mat_nodes["Mix Shader-3-new"].inputs[0].default_value = 0.5

            mat_links.new(mat_nodes["ColorRamp"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])
            mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-1-new"].inputs[20]) 
            mat_links.new(mat_nodes["Bump.001"].outputs[0], mat_nodes["Principled BSDF-2-new"].inputs[20])   

            mat_links.new(mat_nodes["Principled BSDF"].outputs[0], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-new"].outputs[0], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Mix Shader"].inputs[1])

            mat_links.new(mat_nodes["Principled BSDF.001"].outputs[0], mat_nodes["Mix Shader-1-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-1-new"].outputs[0], mat_nodes["Mix Shader-1-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-1-new"].outputs[0], mat_nodes["Mix Shader"].inputs[2])

            mat_links.new(mat_nodes["Principled BSDF.002"].outputs[0], mat_nodes["Mix Shader-2-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-2-new"].outputs[0], mat_nodes["Mix Shader-2-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-2-new"].outputs[0], mat_nodes["Mix Shader.001"].inputs[1])

            mat_links.new(mat_nodes["Principled BSDF.003"].outputs[0], mat_nodes["Mix Shader-3-new"].inputs[1])
            mat_links.new(mat_nodes["Principled BSDF-3-new"].outputs[0], mat_nodes["Mix Shader-3-new"].inputs[2])
            mat_links.new(mat_nodes["Mix Shader-3-new"].outputs[0], mat_nodes["Mix Shader.001"].inputs[2])        
    elif material_name == "metal_9":
        ## Random Parameters
        # mat_nodes["Principled BSDF"].inputs[4].default_value = random.uniform(0.98, 1.00)       # metallic
        # mat_nodes["Principled BSDF"].inputs[5].default_value = random.uniform(0.5, 1.0)         # specular
        # mat_nodes["Principled BSDF"].inputs[6].default_value = random.uniform(0.0, 1.0)         # specularTint
        mat_nodes["Principled BSDF"].inputs[7].default_value = random.uniform(0.01, 0.3)         # roughness
        # mat_nodes["Principled BSDF"].inputs[12].default_value = random.uniform(0.0, 0.3)        # clearcoat
        # mat_nodes["Principled BSDF"].inputs[13].default_value = random.uniform(0.0, 0.3)        # clearcoatGloss
        mat_nodes["Anisotropic BSDF"].inputs[1].default_value = random.uniform(0.11, 0.25)
        mat_nodes["Anisotropic BSDF"].inputs[2].default_value = random.uniform(0.4, 0.6)

        if transfer_flag == True:
            bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_new.name = 'Principled BSDF-new'
            for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
                bsdf_new.inputs[key].default_value = input.default_value

            mix_new = mat_nodes.new(type='ShaderNodeMixShader')
            mix_new.name = 'Mix Shader-new'

            if is_texture:
                mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
                mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9
                mat_links.new(tex_node.outputs[0], mat_nodes["Anisotropic BSDF"].inputs[0])
            else:
                mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
                mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9
                mat_nodes["Anisotropic BSDF"].inputs[0].default_value = list(orign_base_color)

            mat_links.new(mat_nodes["Principled BSDF-new"].outputs[0], mat_nodes["Mix Shader-new"].inputs[2])
            mat_links.new(mat_nodes["Principled BSDF"].outputs[0], mat_nodes["Mix Shader-new"].inputs[1])
            mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Mix Shader"].inputs[1])

    ## porcelain
    elif material_name == "porcelain_0":
        # if is_texture:
        #     mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        # else:
        #     mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
        bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_new.name = 'Principled BSDF-new'
        for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
            bsdf_new.inputs[key].default_value = input.default_value

        mix_new = mat_nodes.new(type='ShaderNodeMixShader')
        mix_new.name = 'Mix Shader-new'

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
            mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9   
        else:
            mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Mix Shader-new"].inputs[0].default_value = mix_prop#0.8   

        mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
        mat_links.new(mat_nodes["Image Texture.001"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

        mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
        mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
        mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])    
    elif material_name == "porcelain_1":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Mix"].inputs[1])
        else:
            mat_nodes["Mix"].inputs[1].default_value = list(orign_base_color)  
    elif material_name == "porcelain_2":
        bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_new.name = 'Principled BSDF-new'
        for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
            bsdf_new.inputs[key].default_value = input.default_value

        mix_new = mat_nodes.new(type='ShaderNodeMixShader')
        mix_new.name = 'Mix Shader-new'

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
            mat_nodes["Mix Shader-new"].inputs[0].default_value = tex_mix_prop#0.9   
        else:
            mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.8   

        mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
        mat_links.new(mat_nodes["Image Texture.001"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

        mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
        mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
        mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])    
    elif material_name == "porcelain_3":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Mix.001"].inputs[1])
        else:
            mat_nodes["Mix.001"].inputs[1].default_value = list(orign_base_color)   
    elif material_name == "porcelain_4":
        # if is_texture:
        #     mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF"].inputs[0])
        #     mat_links.new(tex_node.outputs[0], mat_nodes["Glossy BSDF"].inputs[0])
        # else:
        #     mat_nodes["Diffuse BSDF"].inputs[0].default_value = list(orign_base_color)
        #     mat_nodes["Glossy BSDF"].inputs[0].default_value = list(orign_base_color)
        mat_nodes["Glossy BSDF"].inputs[1].default_value = random.uniform(0.05, 0.15)

        diff_new = mat_nodes.new(type='ShaderNodeBsdfDiffuse')
        diff_new.name = 'Diffuse BSDF-new'
        for key, input in enumerate(mat_nodes["Diffuse BSDF"].inputs):
            diff_new.inputs[key].default_value = input.default_value

        mix_new = mat_nodes.new(type='ShaderNodeMixShader')
        mix_new.name = 'Mix Shader-new'

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF-new"].inputs[0])
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0   
        else:
            mat_nodes["Diffuse BSDF"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9   

        mat_links.new(mat_nodes["Diffuse BSDF"].outputs[0], mat_nodes["Mix Shader-new"].inputs[1])
        mat_links.new(mat_nodes["Diffuse BSDF-new"].outputs[0], mat_nodes["Mix Shader-new"].inputs[2])
        mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Mix Shader"].inputs[1])
    elif material_name == "porcelain_5":
        # if is_texture:
        #     mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF"].inputs[0])
        #     mat_links.new(tex_node.outputs[0], mat_nodes["Glossy BSDF"].inputs[0])
        # else:
        #     mat_nodes["Diffuse BSDF"].inputs[0].default_value = list(orign_base_color)
        #     mat_nodes["Glossy BSDF"].inputs[0].default_value = list(orign_base_color)
        diff_new = mat_nodes.new(type='ShaderNodeBsdfDiffuse')
        diff_new.name = 'Diffuse BSDF-new'
        for key, input in enumerate(mat_nodes["Diffuse BSDF"].inputs):
            diff_new.inputs[key].default_value = input.default_value

        mix_new = mat_nodes.new(type='ShaderNodeMixShader')
        mix_new.name = 'Mix Shader-new'

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF-new"].inputs[0])
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0   
        else:
            mat_nodes["Diffuse BSDF"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9   

        mat_links.new(mat_nodes["Diffuse BSDF"].outputs[0], mat_nodes["Mix Shader-new"].inputs[1])
        mat_links.new(mat_nodes["Diffuse BSDF-new"].outputs[0], mat_nodes["Mix Shader-new"].inputs[2])
        mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Mix Shader"].inputs[1])
    elif material_name == "porcelain_6":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF"].inputs[0])
        else:
            mat_nodes["Diffuse BSDF"].inputs[0].default_value = list(orign_base_color)  
    
    ## plastic
    elif material_name == "plastic_1":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF.001"].inputs[0])
        else:
            mat_nodes["Principled BSDF.001"].inputs[0].default_value = list(orign_base_color)            
    elif material_name == "plastic_2":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF.001"].inputs[0])
        else:
            mat_nodes["Principled BSDF.001"].inputs[0].default_value = list(orign_base_color)    
    elif material_name == "plastic_3":
        mat_nodes["值(明度)"].outputs[0].default_value = random.uniform(0.05, 0.25)

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF"].inputs[0])
        else:
            mat_nodes["Diffuse BSDF"].inputs[0].default_value = list(orign_base_color)  
    elif material_name == "plastic_5":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)    
    elif material_name == "plastic_6":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Reroute.012"].inputs[0])    
            mat_links.new(tex_node.outputs[0], mat_nodes["Reroute.021"].inputs[0])    
            mat_links.new(tex_node.outputs[0], mat_nodes["Reroute.022"].inputs[0])    
            mat_links.new(tex_node.outputs[0], mat_nodes["Reroute.033"].inputs[0])  
        else:
            mat_nodes["RGB"].outputs[0].default_value = list(orign_base_color)
            mat_nodes["RGB.001"].outputs[0].default_value = list(orign_base_color)
            """
            mat_nodes["RGB.002"].outputs[0].default_value = list(orign_base_color)
            mat_nodes["RGB.003"].outputs[0].default_value = list(orign_base_color)
            """

    ## rubber
    elif material_name == "rubber_0":
        diff_new = mat_nodes.new(type='ShaderNodeBsdfDiffuse')
        diff_new.name = 'Diffuse BSDF-new'
        for key, input in enumerate(mat_nodes["Diffuse BSDF"].inputs):
            diff_new.inputs[key].default_value = input.default_value

        mix_new = mat_nodes.new(type='ShaderNodeMixShader')
        mix_new.name = 'Mix Shader-new'

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF-new"].inputs[0])
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0   
        else:
            mat_nodes["Diffuse BSDF"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9   

        mat_links.new(mat_nodes["Diffuse BSDF"].outputs[0], mat_nodes["Mix Shader-new"].inputs[1])
        mat_links.new(mat_nodes["Diffuse BSDF-new"].outputs[0], mat_nodes["Mix Shader-new"].inputs[2])
        mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Mix Shader"].inputs[1])
    elif material_name == "rubber_1":
        bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_new.name = 'Principled BSDF-new'
        for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
            bsdf_new.inputs[key].default_value = input.default_value

        mix_new = mat_nodes.new(type='ShaderNodeMixShader')
        mix_new.name = 'Mix Shader-new'

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0  
        else:
            mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9  

        mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
        mat_links.new(mat_nodes["RGB Curves"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

        mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
        mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
        mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
    elif material_name == "rubber_2":
        bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_new.name = 'Principled BSDF-new'
        for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
            bsdf_new.inputs[key].default_value = input.default_value

        mix_new = mat_nodes.new(type='ShaderNodeMixShader')
        mix_new.name = 'Mix Shader-new'

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0  
        else:
            mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9  

        mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
        mat_links.new(mat_nodes["RGB Curves"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

        mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
        mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
        mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
    elif material_name == "rubber_3":
        bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_new.name = 'Principled BSDF-new'
        for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
            bsdf_new.inputs[key].default_value = input.default_value

        mix_new = mat_nodes.new(type='ShaderNodeMixShader')
        mix_new.name = 'Mix Shader-new'

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0  
        else:
            mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9  

        mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
        mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
        mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
    elif material_name == "rubber_4":
        bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_new.name = 'Principled BSDF-new'
        for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
            bsdf_new.inputs[key].default_value = input.default_value

        mix_new = mat_nodes.new(type='ShaderNodeMixShader')
        mix_new.name = 'Mix Shader-new'

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0  
        else:
            mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9  

        mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
        mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
        mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
    
    ## plastic_specular
    elif material_name == "plasticsp_0":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Reroute.001"].inputs[0])
            mat_links.new(tex_node.outputs[0], mat_nodes["Reroute"].inputs[0])
        else:
            mat_nodes["RGB.001"].outputs[0].default_value = list(orign_base_color)
    elif material_name == "plasticsp_1":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
      
    ## paint_specular
    elif material_name == "paintsp_0":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            mat_links.new(tex_node.outputs[0], mat_nodes["Diffuse BSDF"].inputs[0])
        else:
            mat_nodes["RGB"].outputs[0].default_value = list(orign_base_color)
    elif material_name == "paintsp_1":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Glossy BSDF"].inputs[0])
            mat_links.new(tex_node.outputs[0], mat_nodes["Mix"].inputs[2])
            mat_links.new(tex_node.outputs[0], mat_nodes["Mix.001"].inputs[1])
            mat_links.new(tex_node.outputs[0], mat_nodes["Hue Saturation Value"].inputs[4])
        else:
            mat_nodes["RGB"].outputs[0].default_value = list(orign_base_color)
    elif material_name == "paintsp_2":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Group"].inputs[0])
        else:
            mat_nodes["Group"].inputs[0].default_value = list(orign_base_color)
    elif material_name == "paintsp_3":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
    elif material_name == "paintsp_4":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
            mat_links.new(tex_node.outputs[0], mat_nodes["Glossy BSDF"].inputs[0])
            mat_links.new(tex_node.outputs[0], mat_nodes["Glossy BSDF.001"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Glossy BSDF"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Glossy BSDF.001"].inputs[0].default_value = list(orign_base_color)
    elif material_name == "paintsp_5":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Invert"].inputs[1])
            mat_links.new(tex_node.outputs[0], mat_nodes["Reroute.002"].inputs[0])
        else:
            mat_nodes["RGB"].outputs[0].default_value = list(orign_base_color)

    ## rubber
    elif material_name == "rubber_5":
        bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_new.name = 'Principled BSDF-new'
        for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
            bsdf_new.inputs[key].default_value = input.default_value

        mix_new = mat_nodes.new(type='ShaderNodeMixShader')
        mix_new.name = 'Mix Shader-new'

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 1.0  
        else:
            mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9  

        mat_links.new(mat_nodes["Mix.005"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7]) 
        mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
        mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
        mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])

    ## plastic
    elif material_name == "plastic_0":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF.001"].inputs[0])
        else:
            mat_nodes["Principled BSDF.001"].inputs[0].default_value = list(orign_base_color)
    elif material_name == "plastic_4":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
    elif material_name == "plastic_7":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF.001"].inputs[0])    
        else:
            mat_nodes["RGB"].outputs[0].default_value = list(orign_base_color)
    elif material_name == "plastic_8":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Group"].inputs[0])    
        else:
            mat_nodes["Group"].inputs[0].default_value = list(orign_base_color)
    elif material_name == "plastic_9":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)            
    elif material_name == "plastic_10":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)            
    elif material_name == "plastic_11":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
    elif material_name == "plastic_12":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Mix"].inputs[2])    
        else:
            mat_nodes["RGB"].outputs[0].default_value = list(orign_base_color)
    elif material_name == "plastic_13":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
    elif material_name == "plastic_14":
        mat_nodes["Math.005"].inputs[1].default_value = random.uniform(0.05, 0.3)

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
    
    ## paper
    elif material_name == "paper_0":
        bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_new.name = 'Principled BSDF-new'
        for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
            bsdf_new.inputs[key].default_value = input.default_value

        mix_new = mat_nodes.new(type='ShaderNodeMixShader')
        mix_new.name = 'Mix Shader-new'

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9  
        else:
            mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Mix Shader-new"].inputs[0].default_value = 0.9  

        mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
        mat_links.new(mat_nodes["Mix.002"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

        mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
        mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
        mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
    elif material_name == "paper_1":
        bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_new.name = 'Principled BSDF-new'
        for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
            bsdf_new.inputs[key].default_value = input.default_value

        mix_new = mat_nodes.new(type='ShaderNodeMixShader')
        mix_new.name = 'Mix Shader-new'

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
            mat_nodes["Mix Shader-new"].inputs[0].default_value = random.uniform(0.8, 0.95)  
        else:
            mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Mix Shader-new"].inputs[0].default_value = random.uniform(0.8, 0.9)  

        mat_links.new(mat_nodes["Normal Map"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
        mat_links.new(mat_nodes["Image Texture.001"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

        mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
        mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
        mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
    elif material_name == "paper_2":
        bsdf_new = mat_nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_new.name = 'Principled BSDF-new'
        for key, input in enumerate(mat_nodes["Principled BSDF"].inputs):
            bsdf_new.inputs[key].default_value = input.default_value

        mix_new = mat_nodes.new(type='ShaderNodeMixShader')
        mix_new.name = 'Mix Shader-new'

        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF-new"].inputs["Base Color"])
            mat_nodes["Mix Shader-new"].inputs[0].default_value = random.uniform(0.9, 0.95)  
        else:
            mat_nodes["Principled BSDF-new"].inputs[0].default_value = list(orign_base_color)
            mat_nodes["Mix Shader-new"].inputs[0].default_value = random.uniform(0.9, 0.95)  

        mat_links.new(mat_nodes["Bump"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[20])
        mat_links.new(mat_nodes["Bright/Contrast"].outputs[0], mat_nodes["Principled BSDF-new"].inputs[7])

        mat_links.new(mat_nodes["Principled BSDF"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[1])
        mat_links.new(mat_nodes["Principled BSDF-new"].outputs["BSDF"], mat_nodes["Mix Shader-new"].inputs[2])
        mat_links.new(mat_nodes["Mix Shader-new"].outputs[0], mat_nodes["Material Output"].inputs["Surface"])
    
    ## leather
    elif material_name == "leather_0":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
    elif material_name == "leather_1":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
    elif material_name == "leather_2":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Mix"].inputs[1])
        else:
            mat_nodes["Mix"].inputs[1].default_value = list(orign_base_color)
    elif material_name == "leather_3":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
    elif material_name == "leather_4":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color)
    elif material_name == "leather_5":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["lether"].inputs[0])
        else:
            mat_nodes["lether"].inputs[0].default_value = list(orign_base_color)

    ## wood
    elif material_name == "wood_0":
        pass
    elif material_name == "wood_1":
        pass
    elif material_name == "wood_2":
        pass
    elif material_name == "wood_3":
        pass
    elif material_name == "wood_4":
        pass
    elif material_name == "wood_5":
        pass
    elif material_name == "wood_6":
        pass
    elif material_name == "wood_7":
        pass
    elif material_name == "wood_8":
        pass
    elif material_name == "wood_9":
        pass

    ## fabric
    elif material_name == "fabric_0":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
    elif material_name == "fabric_1":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
    elif material_name == "fabric_2":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Mix"].inputs[1])
        else:
            mat_nodes["Mix"].inputs[1].default_value = list(orign_base_color) 

    ## clay
    elif material_name == "clay_0":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
    elif material_name == "clay_1":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
    elif material_name == "clay_2":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
    elif material_name == "clay_3":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Mix"].inputs[1])
        else:
            mat_nodes["Mix"].inputs[1].default_value = list(orign_base_color) 
    elif material_name == "clay_4":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Principled BSDF"].inputs[0])
        else:
            mat_nodes["Principled BSDF"].inputs[0].default_value = list(orign_base_color) 
    elif material_name == "clay_5":
        if is_texture:
            mat_links.new(tex_node.outputs[0], mat_nodes["Mix"].inputs[1])
        else:
            mat_nodes["Mix"].inputs[1].default_value = list(orign_base_color) 

    ## glass
    elif material_name == "glass_0":
        mat_nodes["Mix Shader"].inputs[0].default_value = random.uniform(0.1, 0.3)
        mat_nodes["Glossy BSDF"].inputs[1].default_value = random.uniform(0.1, 0.3)
    elif material_name == "glass_4":
        mat_nodes["Layer Weight"].inputs[0].default_value = random.uniform(0.3, 0.7)
        mat_nodes["Glossy BSDF"].inputs[1].default_value = random.uniform(0.05, 0.2)
    elif material_name == "glass_5":
        mat_nodes["Layer Weight"].inputs[0].default_value = random.uniform(0.2, 0.4)
        mat_nodes["Glossy BSDF"].inputs[1].default_value = random.uniform(0.0, 0.1)
    elif material_name == "glass_14":
        mat_nodes["Glass BSDF.005"].inputs[1].default_value = random.uniform(0.0, 0.1)
        mat_nodes["Glass BSDF.006"].inputs[1].default_value = random.uniform(0.0, 0.1)
        mat_nodes["Glass BSDF.007"].inputs[1].default_value = random.uniform(0.0, 0.1)
        mat_nodes["Glass BSDF.008"].inputs[1].default_value = random.uniform(0.0, 0.1)
        mat_nodes["Layer Weight"].inputs[0].default_value = random.uniform(0.81, 0.87)
        mat_nodes["Layer Weight.001"].inputs[0].default_value = random.uniform(0.65, 0.71)
        mat_nodes["Layer Weight.002"].inputs[0].default_value = random.uniform(0.81, 0.87)
        color_value = random.uniform(0.599459, 0.70)
        mat_nodes["Transparent BSDF"].inputs[0].default_value = list([color_value, color_value, color_value, 1])

    # elif material_name == "glass_15":
    #     mat_nodes["Glass BSDF"].inputs[1].default_value = random.uniform(0.0, 0.1)
    #     mat_nodes["Glass BSDF"].inputs[2].default_value = random.uniform(1.325, 1.335)
    #     color_value = random.uniform(0.297, 0.35)
    #     mat_nodes["Transparent BSDF"].inputs[0].default_value = list([color_value, color_value, color_value, 1])

    else:
        print(material_name + " no change")



def set_modify_material(obj, material):
    for mat_slot in obj.material_slots:
        #print("-material_slots:",mat_slot)
        if mat_slot.material:
            if mat_slot.material.node_tree:
                #print("--material:" + str(mat_slot.material.name))
                # Copy one material from the library
                srcmat = material
                #print(srcmat.name)
                mat = srcmat.copy()
                mat.name = mat_slot.material.name   # rename
                mat_links = mat.node_tree.links
                mat_nodes = mat.node_tree.nodes
                # Obtain the Principled BSDF node
                bsdf_node = mat_slot.material.node_tree.nodes.get("Principled BSDF", None)
                if bsdf_node is not None:
                    # Obtain the Texture node, and determine w or w/o texture map
                    tex_node_orign = mat_slot.material.node_tree.nodes.get("Image Texture", None)
                    # if the texture map exists
                    if tex_node_orign is not None:
                        #mat = mat_slot.material.copy() 
                        # Get the bl_idname to create a new node of the same type
                        tex_node = mat_nodes.new(tex_node_orign.bl_idname)
                        texture_img = bpy.data.images[tex_node_orign.image.name]
                        # Assign the default values from the old node to the new node
                        tex_node.image = texture_img
                        #tex_node.location = Vector((-800, 0))
                        # create the new link for mat
                        modify_material(mat_links, mat_nodes, srcmat.name, is_texture=True, tex_node=tex_node)
                    # if the texture map does not exist
                    else:
                        # obtain the base color
                        orign_base_color = mat_slot.material.node_tree.nodes["Principled BSDF"].inputs[0].default_value
                        if orign_base_color[0] == 0.0 and orign_base_color[1] == 0.0 and orign_base_color[2] == 0.0:
                            orign_base_color == [0.05, 0.05, 0.05, 1]
                        # apply to mat
                        modify_material(mat_links, mat_nodes, srcmat.name, is_texture=False, orign_base_color=orign_base_color)

                # apply the material
                bpy.data.materials.remove(mat_slot.material)
                mat_slot.material = mat


def set_modify_raw_material(obj):
    for mat_slot in obj.material_slots:
        if mat_slot.material:
            if mat_slot.material.node_tree:
                bsdf_node = mat_slot.material.node_tree.nodes.get("Principled BSDF", None)
                if bsdf_node is not None:
                    # Obtain the Texture node, and determine w or w/o texture map
                    tex_node_orign = mat_slot.material.node_tree.nodes.get("Image Texture", None)
                    # if the texture map does not exist
                    if tex_node_orign is None:
                            # obtain base color
                            orign_base_color = mat_slot.material.node_tree.nodes["Principled BSDF"].inputs[0].default_value
                            if orign_base_color[0] == 0.0 and orign_base_color[1] == 0.0 and orign_base_color[2] == 0.0:
                                mat = mat_slot.material.copy()
                                mat.name = mat_slot.material.name   # rename
                                mat_nodes = mat.node_tree.nodes   
                                mat_nodes["Principled BSDF"].inputs[0].default_value = list([0.05, 0.05, 0.05, 1])
                                # apply the material
                                bpy.data.materials.remove(mat_slot.material)
                                mat_slot.material = mat
