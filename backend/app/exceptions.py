"""
業務異常類
用於 Service 層拋出業務異常，由 API 層轉換為 HTTP 響應
"""


class ImageNotFoundError(Exception):
    """圖像不存在異常"""
    pass


class ImageFileNotFoundError(Exception):
    """圖像文件不存在異常"""
    pass


class InvalidBboxError(Exception):
    """無效的邊界框異常"""
    pass


class InvalidBboxSizeError(Exception):
    """邊界框尺寸太小異常"""
    pass


class InvalidCenterError(Exception):
    """無效的中心點異常"""
    pass


class InvalidSealDataError(Exception):
    """無效的印鑑數據異常"""
    pass


class ImageNotMarkedError(Exception):
    """圖像未標記印鑑位置異常"""
    pass


class ImageReadError(Exception):
    """無法讀取圖像文件異常"""
    pass


class CropAreaTooSmallError(Exception):
    """裁切區域太小異常"""
    pass


class ComparisonNotFoundError(Exception):
    """比對記錄不存在異常"""
    pass


class VisualizationNotFoundError(Exception):
    """視覺化記錄不存在異常"""
    pass


class MultiSealComparisonTaskNotFoundError(Exception):
    """多印鑑比對任務不存在異常"""
    pass

