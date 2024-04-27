from osgeo import gdal
import numpy as np




def read_tiff(inpath):
    gdal.AllRegister()
    ds = gdal.Open(inpath)
    proj = ds.GetProjection()
    col = ds.RasterXSize
    row = ds.RasterYSize
    band = ds.RasterCount
    geoTransform = ds.GetGeoTransform()
    test = ds.GetRasterBand(1).ReadAsArray()
    data = np.zeros([row, col, band])
    for i in range(band):
        dt = ds.GetRasterBand(i+1)
        data[:, :, i] = dt.ReadAsArray()
    return data.astype(test.dtype), geoTransform, proj


def write_tiff(outpath, array, geoTransform, proj):
    array = np.expand_dims(array, axis=2)
    rows, cols, band = array.shape
    driver = gdal.GetDriverByName('Gtiff')
    outRaster = driver.Create(outpath, cols, rows, band, gdal.GDT_Float32, options=[
        "TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])
    outRaster.SetGeoTransform(geoTransform)
    outRaster.SetProjection(proj)
    for i in range(band):
        outRaster.GetRasterBand(i+1).WriteArray(array[..., i])
    outRaster.FlushCache()
    return None

def write_tiff2(outpath, array, geoTransform, proj):
    array = np.expand_dims(array, axis=2)
    rows, cols, band = array.shape
    driver = gdal.GetDriverByName('Gtiff')
    outRaster = driver.Create(outpath, cols, rows, band, gdal.GDT_Byte  , options=[
        "TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])
    outRaster.SetGeoTransform(geoTransform)
    outRaster.SetProjection(proj)
    for i in range(band):
        outRaster.GetRasterBand(i+1).WriteArray(array[..., i])
    outRaster.FlushCache()
    return None


def fill_ndarray(t1):
    for i in range(t1.shape[1]): # 遍历每一列（每一列中的nan替换成该列的均值）
        temp_col = t1[:, i] # 当前的一列
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0: # 不为0，说明当前这一列中有nan
            temp_not_nan_col = temp_col[temp_col == temp_col] # 去掉nan的ndarray
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean() # mean()表示求均值。
    return t1