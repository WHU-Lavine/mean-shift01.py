import gdal
import numpy as np
from skimage import filters, measure, morphology
import matplotlib.pyplot


def waterIndex(band2, band4, band5):
    """计算水体指数，返回混合水体指数，依次为mndwi,0.9mndwi+0.1ndwi,......ndwwi"""
    mndwi = (band2 - band5) / (band2 + band5 + 0.000001)
    ndwi = (band2 - band4) / (band2 + band4 + 0.000001)
    composite_water_index = []
    α = 0
    while α <= 1:
        temp_composite_water_index = α * ndwi + (1 - α) * mndwi
        composite_water_index.append(temp_composite_water_index)
        α += 0.1
    return composite_water_index


def read(filename):
    """读取图像"""
    imgDataset = gdal.Open(filename)
    """imgData = imgDataset.ReadAsArray(0, 0, imgDataset.RasterXSize, imgDataset.RasterYSize)
    imgGeo = imgDataset.GetGeoTransform()
    imgPrj = imgDataset.GetProjection()"""
    return imgDataset


def write(filename, im_proj, im_geotrans, im_data):

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


if __name__ == '__main__':
    dataset = read(r'C:\Users\Administrator\Desktop\new.tiff')
    data = dataset.ReadAsArray(0, 0, dataset.RasterXSize, dataset.RasterYSize).astype(np.float32)

    # collection = data[2]-data[5]

    collection = waterIndex(data[1], data[3], data[4])
    # print(collection)
    # 计算初步全局阈值
    wholeThreshold = filters.threshold_otsu(collection[0])
    print(wholeThreshold)
    wholeWater = (collection[0] >= wholeThreshold)
    labels = measure.label(wholeWater, connectivity=1)
    label_att = measure.regionprops(labels)
    c = label_att[1]
    #matplotlib.pyplot.imshow(wholeWater)
    #matplotlib.pyplot.show()
    resWholeWater = morphology.remove_small_objects(wholeWater, min_size=10, connectivity=1, in_place=False)
    #matplotlib.pyplot.imshow(resWholeWater)
    #matplotlib.pyplot.show()
    print(labels.max()+1)
    write(r'C:\Users\Administrator\Desktop\new.tiff', dataset.GetProjection(), dataset.GetGeoTransform(), resWholeWater)

