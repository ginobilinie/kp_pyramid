def get_architectures(arch):
    arch = arch.lower()
    if (arch == 'pyramid_v1' or arch == 'kp_pyramid_v1' or arch == 'pyramid_v2'
            or arch == 'kp_pyramid_v2'):
        architecture = [[
            'simple', 'resnetb', 'resnetb_strided', 'resnetb',
            'resnetb_strided', 'resnetb', 'resnetb_strided', 'resnetb',
            'resnetb_strided', 'resnetb', 'resnetb_strided', 'resnetb'
        ], ['unary', 'unary', 'unary', 'unary'], ['unary', 'unary', 'unary'],
                        ['unary', 'unary'], ['unary']]
    elif (arch == 'pyramid_v1_deform' or arch == 'kp_pyramid_v1_deform'
          or arch == 'pyramid_v2_deform' or arch == 'kp_pyramid_v2_deform'):
        architecture = [[
            'simple', 'resnetb', 'resnetb_strided', 'resnetb',
            'resnetb_strided', 'resnetb_deformable',
            'resnetb_deformable_strided', 'resnetb_deformable',
            'resnetb_deformable_strided', 'resnetb_deformable',
            'resnetb_deformable_strided', 'resnetb_deformable'
        ], ['unary', 'unary', 'unary', 'unary'], ['unary', 'unary', 'unary'],
                        ['unary', 'unary'], ['unary']]
    elif arch == 'kpconv' or arch == 'kpfcnn':
        architecture = [
            'simple', 'resnetb', 'resnetb_strided', 'resnetb',
            'resnetb_strided', 'resnetb', 'resnetb_strided', 'resnetb',
            'resnetb_strided', 'resnetb', 'nearest_upsample', 'unary',
            'nearest_upsample', 'unary', 'nearest_upsample', 'unary',
            'nearest_upsample', 'unary'
        ]
    elif arch == 'kpconv_deform' or arch == 'kpfcnn_deform':
        architecture = [
            'simple', 'resnetb', 'resnetb_strided', 'resnetb',
            'resnetb_strided', 'resnetb_deformable',
            'resnetb_deformable_strided', 'resnetb_deformable',
            'resnetb_deformable_strided', 'resnetb_deformable',
            'nearest_upsample', 'unary', 'nearest_upsample', 'unary',
            'nearest_upsample', 'unary', 'nearest_upsample', 'unary'
        ]

    return architecture
