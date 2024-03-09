import geopandas as gpd
import numpy as np
import folium

def visualize_predictions(df_test: gpd.GeoDataFrame, 
                          y_pred: np.array, 
                          regions_gdf: gpd.GeoDataFrame):
    
    """
    Function for predictions visualization as hexes on map.
    
    Params:
        df_test (gpd.GeoDatFrame): gdf with set of nodes and labels in the set on which the prediction will be made
        y_pred (np.array): array with predictions for each node in given set
        regions_gdf (gpd.GeoDataFrame): gdf with hex regions with resolution the same as used in model
    Returns:
        correct_hexes (folium.folium.Map): folium map with marked true predictions (green), false negatives (red)
                                            and false positives (orange)
    """
    
    
    df_test['pred'] = y_pred
    df_test = df_test.set_index('node')
    df_test = df_test.rename_axis('region_id')
    confusion_gdf = df_test.merge(regions_gdf[['geometry']], left_index=True, right_index=True, how='left')
    confusion_gdf["is_correct"] = confusion_gdf.label == confusion_gdf.pred
    confusion_gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(data=confusion_gdf, geometry="geometry", crs="EPSG:4326")
    correct_hexes: folium.folium.Map = confusion_gdf[confusion_gdf.is_correct == True].explore(style_kwds=dict(fillColor="green", color="green")) 

    if len(confusion_gdf[(confusion_gdf.label == 0.0) & (confusion_gdf.pred == 1.0)])>0:
        confusion_gdf[
            (confusion_gdf.label == 0.0) & (confusion_gdf.pred == 1.0)
        ].explore(
            m=correct_hexes, style_kwds=dict(fillColor="orange", color="orange")
        )
    else:
        print("No false positives")
    if len(confusion_gdf[
            (confusion_gdf.label == 1.0) & (confusion_gdf.pred == 0.0)
        ])>0:
        confusion_gdf[
            (confusion_gdf.lebel == 1.0) & (confusion_gdf.pred == 0.0)
        ].explore(
            m=correct_hexes, style_kwds=dict(fillColor="red", color="red")
        )
    else: 
        print("No false negatives")
    return correct_hexes