import pydicom as dicom
import pandas as pd
import swifter
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm


FIELDS = [
    #'AccessionNumber',                 # duplicate of PatientID
    'AcquisitionMatrix',                
    #'B1rms',                           # nan
    #'BitsAllocated',                   # const: 16
    #'BitsStored',                      # const: 16
    'Columns',                          # 
    #'ConversionType',                  # nan, but rarely WSD
    #'DiffusionBValue',                 # nan, but rarely 0
    #'DiffusionGradientOrientation',    # nan, but rarely [0.0, 0.0, 0.0] 
    'EchoNumbers',                      # http://mriquestions.com/fse-parameters.html
    #'EchoTime',                        # nan
    'EchoTrainLength',                  # http://mriquestions.com/fse-parameters.html
    'FlipAngle',                        # http://mriquestions.com/what-is-flip-angle.html
    #'HighBit',                         # const: 15
    'HighRRValue',                      # nan but rarely 0
    #'ImageDimensions',                 # nan but rarely 2
    #'ImageFormat',                     # nan but rarely RECT
    #'ImageGeometryType',               # nan but rarely PLANAR
    #'ImageLocation',                   # nan but rarely 32736.0
    #'ImageOrientation',                # nan but rarely something else
    'ImageOrientationPatient',          # see notebook
    #'ImagePosition',                   # nan but rarely something else
    'ImagePositionPatient',             # see notebook
    #'ImageType',                       # const: ['DERIVED', 'SECONDARY']            
    #'ImagedNucleus',                   # const: 1H 
    'ImagingFrequency',                 # const: near 0
    'InPlanePhaseEncodingDirection',    # usually row, sometimes col
    'InStackPositionNumber',            # position of image in series
    'InstanceNumber',                   # image ID
    #'InversionTime',                   # nan
    #'Laterality',                      # nan
    #'LowRRValue',                      # nan but rarely 0
    'MRAcquisitionType',                # 3D but rarely 2D
    'MagneticFieldStrength',            # 3 but rarely 1.5
    #'Modality',                        # const: MR
    'NumberOfAverages',                 # 1-3
    'NumberOfPhaseEncodingSteps',
    'PatientID',
    #'PatientName',                     # duplicate of PatientID
    #'PatientPosition',                 # const: HFS
    'PercentPhaseFieldOfView',
    'PercentSampling',
    # 'PhotometricInterpretation',      # const: MONOCHROME2
    'PixelBandwidth',
    'PixelPaddingValue',
    'PixelRepresentation',
    'PixelSpacing',
    'PlanarConfiguration',
    #'PositionReferenceIndicator',      # nan
    'PresentationLUTShape',
    'ReconstructionDiameter',
    #'RescaleIntercept',                # const: 0.0
    #'RescaleSlope',                    # const: 1.0
    #'RescaleType',                     # const: US    'Rows',
    'SAR',
    #'SOPClassUID',                     # const: 1.2.840.10008.5.1.4.1.1.4
    'SOPInstanceUID',
    #'SamplesPerPixel',                 # const: 1
    'SeriesDescription',                # FLAIR, etc... 
    'SeriesInstanceUID',
    'SeriesNumber',                     # see notebook
    'SliceLocation',
    'SliceThickness',
    'SpacingBetweenSlices',             # const: 1
    'SpatialResolution',
    'SpecificCharacterSet',
    #'StudyInstanceUID',                # nan
    #'TemporalResolution',
    #'TransferSyntaxUID',               # const: 1.2.840.10008.1.2
    'TriggerWindow',
    'WindowCenter',
    'WindowWidth'
]

SERIES_TYPES = ['FLAIR', 'T1wCE', 'T1w', 'T2w']


def get_dicom_files(labels):
    dicom_files = []
    for patient_id in labels.str_id:
        for file_type in SERIES_TYPES:
            file_path = f'data/train/{patient_id}/{file_type}/*.dcm'
            dicom_files += glob(file_path)
    return dicom_files


def get_meta_info(dicom_path):
    dicom_file = dicom.read_file(dicom_path, force=True)
    row = {f: dicom_file.get(f) for f in FIELDS}
    row['image_plane'] = get_image_plane(row['ImageOrientationPatient'])
    row['image_position_x'], \
        row['image_position_y'], \
        row['image_position_z'] = row['ImagePositionPatient']
    return row


def get_image_plane(loc):
    # conver listof floats into 1 of 3 planes
    loc = map(round, loc)
    row_x, row_y, row_z, col_x, col_y, col_z = loc
    if row_x == 1 and row_y == 0 and col_x == 0 and col_y == 0:
        return "Coronal"
    elif row_x == 0 and row_y == 1 and col_x == 0 and col_y == 0:
        return "Sagittal"
    elif row_x == 1 and row_y == 0 and col_x == 0 and col_y == 1:
        return "Axial"
    else:
        return "Unknown"


def create_meta_df(dicom_files):
    with Pool(14) as p:
        meta_dicts = list(tqdm(p.imap(get_meta_info, dicom_files), total=len(dicom_files)))
    # convert list of dicts to pandas df
    return pd.json_normalize(meta_dicts)


def get_label_df(data_dir):
    labels = pd.read_csv(f'{data_dir}/train_labels.csv')
    labels['str_id'] = labels.BraTS21ID.astype(str).str.zfill(5)
    return labels


def main(data_dir='./data/'):
    labels = get_label_df(data_dir)
    dicom_files = get_dicom_files(labels)
    meta_df = create_meta_df(dicom_files)
    meta_df.to_csv(data_dir + 'train_metadata.csv', index=False)


if __name__ == "__main__":
    main()