import os

def getAbsPath(path):
    return os.path.abspath(path)

DIR_BIN_PIECEREG = '/home/ethan/Research/PieceReg/PieceReg_ToG/build/bin'
Dir_Build_OpenMVG = ''
Dir_Build_OpenMVS = ''

# sfm
PATH_SFM_PIPELINE   = f'{Dir_Build_OpenMVG}/software/SfM/SfM_SequentialPipeline_FocalLength.py'
BIN_SFM_MVG2MVS     = f'{Dir_Build_OpenMVG}/Linux-x86_64-RELEASE/openMVG_main_openMVG2openMVS'

# openMVS
BIN_DENSE_RECONSTRUCTION = f'{Dir_Build_OpenMVS}/bin/DensifyPointCloud'
BIN_EXPORT_DATA = f'{Dir_Build_OpenMVS}/bin/ExportData'
DIR_BIN_OPENMVS = f'{Dir_Build_OpenMVS}/bin'

## Reg path
BIN_EXTRACT_PIECES = getAbsPath(DIR_BIN_PIECEREG+"/registration/preprocess_ExtractPieces_NoSample")
BIN_PIECES_MATCHING = getAbsPath(DIR_BIN_PIECEREG+"/registration/preprocess_PiecesMatching")
BIN_EXTRACT_BOUNDARY = getAbsPath(DIR_BIN_PIECEREG+"/registration/preprocess_Extract3DBoundaryAndRefine")
BIN_FRONT_BACK_REGISTRATION =  getAbsPath(DIR_BIN_PIECEREG+"/registration/ICPReg")
BIN_DIR = f'{DIR_BIN_PIECEREG}/registration'

# Compile FGR and copy to bin
# https://github.com/isl-org/FastGlobalRegistration
BIN_PATH_FGR = f"{DIR_BIN_PIECEREG}/evaluation/FastGlobalRegistration"