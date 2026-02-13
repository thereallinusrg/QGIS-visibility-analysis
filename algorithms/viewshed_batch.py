# -*- coding: utf-8 -*-

"""
/***************************************************************************
ViewshedAnalysis - Batch Viewshed from AOI
A QGIS plugin
begin : 2026-02-13
copyright : (C) 2026
email : /
***************************************************************************/

/***************************************************************************
* *
* This program is free software; you can redistribute it and/or modify *
* it under the terms of the GNU General Public License as published by *
* the Free Software Foundation version 2 of the License, or *
* any later version. *
* *
***************************************************************************/
"""

from os import path
import math

try:
    from PyQt5.QtCore import QCoreApplication
except ImportError:
    from PyQt6.QtCore import QCoreApplication

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsProcessingException,
    QgsMessageLog,
    QgsGeometry,
    QgsPointXY,
    QgsCoordinateTransform,
    QgsProject,
)

from .modules import visibility as ws
from .modules import Raster as rst

import numpy as np
import time


def _compute_aoi_parameters(point_x, point_y, aoi_geometry):
    """
    For a given observer point and an AOI polygon geometry, compute:
      - azim_1, azim_2 : start and stop azimuth (degrees, 0=North, clockwise)
                         that fully covers the AOI from the observer.
      - radius (max view) : distance from observer to the farthest point of AOI.
      - radius_in (inner radius) : distance from observer to the nearest point of AOI.

    The AOI must be completely covered by the resulting viewshed sector.
    """

    # Extract all vertices from the AOI polygon (including any rings)
    vertices = _extract_vertices(aoi_geometry)

    if not vertices:
        raise QgsProcessingException(
            "AOI polygon has no vertices!"
        )

    # Also sample points along edges for better coverage of curved boundaries
    # (polygon edges are straight, but azimuth coverage of long edges
    #  requires intermediate points)
    edge_samples = _sample_polygon_edges(aoi_geometry, interval=50)
    all_points = vertices + edge_samples

    # Compute distances and bearings from observer to every point
    bearings = []
    distances = []

    for vx, vy in all_points:
        dx = vx - point_x
        dy = vy - point_y
        distance = math.sqrt(dx * dx + dy * dy)
        distances.append(distance)

        # Bearing: 0=North, clockwise
        # math.atan2 gives angle from positive X axis, counter-clockwise
        # Convert: bearing = 90 - degrees(atan2(dy, dx)), then normalise to [0,360)
        angle_deg = math.degrees(math.atan2(dx, dy))  # atan2(east, north) = bearing
        if angle_deg < 0:
            angle_deg += 360.0
        bearings.append(angle_deg)

    # Also compute nearest distance to the polygon boundary (not just vertices)
    observer_point = QgsGeometry.fromPointXY(QgsPointXY(point_x, point_y))
    min_distance = observer_point.distance(aoi_geometry)

    max_distance = max(distances)

    # Determine azimuth range that covers all bearing angles
    azim_1, azim_2 = _compute_azimuth_range(bearings)

    return azim_1, azim_2, max_distance, min_distance


def _extract_vertices(geometry):
    """Extract all vertices from a QgsGeometry as a list of (x, y) tuples."""
    vertices = []
    # Handle multi-polygon and polygon
    if geometry.isMultipart():
        multi_poly = geometry.asMultiPolygon()
        for polygon in multi_poly:
            for ring in polygon:
                for point in ring:
                    vertices.append((point.x(), point.y()))
    else:
        polygon = geometry.asPolygon()
        for ring in polygon:
            for point in ring:
                vertices.append((point.x(), point.y()))
    return vertices


def _sample_polygon_edges(geometry, interval=50):
    """
    Sample additional points along the polygon boundary at a given interval (meters).
    This ensures long edges are properly covered in azimuth calculations.
    """
    try:
        densified = geometry.densifyByDistance(interval)
        return _extract_vertices(densified)
    except:
        return []


def _compute_azimuth_range(bearings):
    """
    Given a list of bearing angles (0-360, North=0, clockwise),
    find the minimal arc [azim_1, azim_2] (clockwise from azim_1 to azim_2)
    that covers all bearings.

    If the points surround the observer (span > 360 effectively), 
    return (0, 360) for a full circle.
    """
    if not bearings:
        return 0.0, 360.0

    # Sort bearings
    sorted_b = sorted(set(round(b, 6) for b in bearings))

    if len(sorted_b) <= 1:
        # Single bearing: create a small wedge
        b = sorted_b[0]
        return (b - 1) % 360, (b + 1) % 360

    # Find the largest gap between consecutive bearings.
    # The sector that does NOT contain that gap is the minimal covering arc.
    n = len(sorted_b)
    max_gap = 0.0
    max_gap_idx = 0

    for i in range(n):
        next_i = (i + 1) % n
        if next_i == 0:
            gap = (sorted_b[0] + 360.0) - sorted_b[-1]
        else:
            gap = sorted_b[next_i] - sorted_b[i]

        if gap > max_gap:
            max_gap = gap
            max_gap_idx = i

    if max_gap < 1.0:
        # Nearly full circle
        return 0.0, 360.0

    # azim_1 is the bearing right after the gap (start of the arc)
    # azim_2 is the bearing right before the gap (end of the arc)
    azim_1_idx = (max_gap_idx + 1) % n
    azim_2_idx = max_gap_idx

    azim_1 = sorted_b[azim_1_idx]
    azim_2 = sorted_b[azim_2_idx]

    # Add a small padding (2 degrees) to ensure full coverage
    azim_1 = (azim_1 - 2.0) % 360.0
    azim_2 = (azim_2 + 2.0) % 360.0

    return round(azim_1, 5), round(azim_2, 5)


class ViewshedBatch(QgsProcessingAlgorithm):

    DEM = 'DEM'
    AOI = 'AOI'
    OBSERVER_POINTS = 'OBSERVER_POINTS'
    OBSERVER_HEIGHT = 'OBSERVER_HEIGHT'

    USE_CURVATURE = 'USE_CURVATURE'
    REFRACTION = 'REFRACTION'
    ANALYSIS_TYPE = 'ANALYSIS_TYPE'

    OUTPUT_DIR = 'OUTPUT_DIR'

    TYPES = ['Binary viewshed', 'Depth below horizon', 'Horizon']

    def __init__(self):
        super().__init__()

    def initAlgorithm(self, config):

        self.addParameter(QgsProcessingParameterFeatureSource(
            self.OBSERVER_POINTS,
            self.tr('Observer location(s)'),
            [QgsProcessing.TypeVectorPoint]))

        self.addParameter(QgsProcessingParameterRasterLayer(
            self.DEM,
            self.tr('Digital elevation model')))

        self.addParameter(QgsProcessingParameterFeatureSource(
            self.AOI,
            self.tr('Area of Interest (polygon)'),
            [QgsProcessing.TypeVectorPolygon]))

        self.addParameter(QgsProcessingParameterNumber(
            self.OBSERVER_HEIGHT,
            self.tr('Observer height (meters)'),
            QgsProcessingParameterNumber.Double,
            defaultValue=1.6, minValue=0))

        self.addParameter(QgsProcessingParameterNumber(
            self.ANALYSIS_TYPE,
            self.tr('Analysis type (0=Binary, 1=Depth below horizon, 2=Horizon)'),
            QgsProcessingParameterNumber.Integer,
            defaultValue=0, minValue=0, maxValue=2))

        self.addParameter(QgsProcessingParameterBoolean(
            self.USE_CURVATURE,
            self.tr('Take in account Earth curvature'),
            False))

        self.addParameter(QgsProcessingParameterNumber(
            self.REFRACTION,
            self.tr('Atmospheric refraction'),
            QgsProcessingParameterNumber.Double,
            defaultValue=0.13, minValue=0, maxValue=1))

        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUTPUT_DIR,
            self.tr('Output folder for viewshed rasters')))

    def shortHelpString(self):
        curr_dir = path.dirname(path.realpath(__file__))
        h = (f"""
            <b>Batch Viewshed from AOI</b>
            <p>
            Automatically creates viewshed rasters for multiple observer points.
            For each observer point, the start/stop azimuth, maximum radius and
            inner radius are automatically computed from the given Area of Interest
            (AOI) polygon, so that the AOI is completely covered by the viewshed.
            </p>
            <p>One output raster is produced per observer point.</p>

            <h3>Parameters</h3>
            <ul>
                <li><em>Observer locations</em>: input point layer with observer positions.</li>
                <li><em>Digital elevation model</em>: DEM raster in a projected (metric) CRS.</li>
                <li><em>Area of Interest</em>: polygon layer defining the area that must
                    be fully covered by each viewshed.</li>
                <li><em>Observer height</em>: height of the observer above ground, in meters.</li>
                <li><em>Analysis type</em>: 0 = Binary viewshed, 1 = Depth below horizon, 2 = Horizon.</li>
                <li><em>Earth curvature</em>: whether to account for Earth curvature.</li>
                <li><em>Atmospheric refraction</em>: refraction coefficient (0-1).</li>
                <li><em>Output folder</em>: directory where output rasters will be saved.</li>
            </ul>

            <h3>How it works</h3>
            <ol>
                <li>For each observer point, the algorithm calculates the azimuth sector,
                    maximum radius (outer distance) and minimum radius (inner distance)
                    from the point to the AOI polygon.</li>
                <li>A viewshed analysis is then performed with these parameters.</li>
                <li>The result is saved as a GeoTIFF in the output folder,
                    named <code>viewshed_&lt;point_id&gt;.tif</code>.</li>
            </ol>

            If you find this tool useful, consider to :

            <a href='https://ko-fi.com/D1D41HYSW' target='_blank'><img height='30' style='border:0px;height:36px;' src='{curr_dir}/kofi2.webp' /></a>

            <b>This GIS tool is intended for peaceful use !</b>
        """)
        return h

    def processAlgorithm(self, parameters, context, feedback):

        raster = self.parameterAsRasterLayer(parameters, self.DEM, context)
        observers = self.parameterAsSource(parameters, self.OBSERVER_POINTS, context)
        aoi_source = self.parameterAsSource(parameters, self.AOI, context)
        observer_height = self.parameterAsDouble(parameters, self.OBSERVER_HEIGHT, context)
        analysis_type = self.parameterAsInt(parameters, self.ANALYSIS_TYPE, context)
        useEarthCurvature = self.parameterAsBool(parameters, self.USE_CURVATURE, context)
        refraction = self.parameterAsDouble(parameters, self.REFRACTION, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        precision = 1  # Normal precision, matching the standard viewshed tool

        # ---- Validate inputs ----
        if raster.crs().mapUnits() != 0:
            err = "\n ****** \n ERROR! \n Raster data has to be projected in a metric system!"
            feedback.reportError(err, fatalError=True)
            raise QgsProcessingException(err)

        if round(abs(raster.rasterUnitsPerPixelX()), 2) != round(abs(raster.rasterUnitsPerPixelY()), 2):
            err = ("\n ****** \n ERROR! \n Raster pixels are irregular in shape "
                   "(probably due to incorrect projection)!")
            feedback.reportError(err, fatalError=True)
            raise QgsProcessingException(err)

        # ---- Collect and reproject the AOI geometry ----
        aoi_features = list(aoi_source.getFeatures())
        if not aoi_features:
            err = "\n ****** \n ERROR! \n No AOI features found!"
            feedback.reportError(err, fatalError=True)
            raise QgsProcessingException(err)

        # Combine all AOI features into a single geometry
        aoi_geom = aoi_features[0].geometry()
        for feat in aoi_features[1:]:
            aoi_geom = aoi_geom.combine(feat.geometry())

        # Reproject AOI to raster CRS if needed
        if aoi_source.sourceCrs() != raster.crs():
            transform = QgsCoordinateTransform(
                aoi_source.sourceCrs(), raster.crs(), QgsProject.instance())
            aoi_geom.transform(transform)

        # ---- Set up coordinate transform for observer points ----
        need_reproject = observers.sourceCrs() != raster.crs()
        if need_reproject:
            obs_transform = QgsCoordinateTransform(
                observers.sourceCrs(), raster.crs(), QgsProject.instance())

        # ---- Open the DEM ----
        raster_path = raster.source()

        # ---- Process each observer point ----
        features = list(observers.getFeatures())
        total = len(features)

        if total == 0:
            err = "\n ****** \n ERROR! \n No observer points found!"
            feedback.reportError(err, fatalError=True)
            raise QgsProcessingException(err)

        start = time.process_time()
        report = []
        output_files = []

        for idx, feat in enumerate(features):
            if feedback.isCanceled():
                break

            feedback.pushInfo(f"\n--- Processing point {idx + 1} of {total} ---")

            geom = feat.geometry()
            pt = geom.asPoint()

            # Reproject point to raster CRS if needed
            if need_reproject:
                pt = obs_transform.transform(pt)

            point_x, point_y = pt.x(), pt.y()

            # Try to get a sensible ID for the output filename
            try:
                point_id = feat["ID"]
            except:
                point_id = feat.id()

            # ---- Compute parameters from AOI ----
            try:
                azim_1, azim_2, max_radius, min_radius = _compute_aoi_parameters(
                    point_x, point_y, aoi_geom)
            except Exception as e:
                feedback.reportError(
                    f"Error computing AOI parameters for point {point_id}: {str(e)}")
                continue

            feedback.pushInfo(
                f"  Point {point_id}: azimuth {azim_1:.1f}-{azim_2:.1f}, "
                f"radius {min_radius:.1f}-{max_radius:.1f} m")

            # ---- Prepare output path ----
            output_path = path.join(output_dir, f"viewshed_{point_id}.tif")
            output_files.append(output_path)

            # ---- Create Raster object for this point ----
            dem = rst.Raster(raster_path, output=output_path)

            # Radius in pixels
            radius_pix = max_radius / dem.pix
            radius_in_pix = min_radius / dem.pix

            # Build a pseudo-point dict matching the format used by ws.viewshed_raster
            pix_coord = dem.pixel_coords(point_x, point_y)

            pt_dict = {
                "id": point_id,
                "z": observer_height,
                "radius": radius_pix,
                "radius_in": radius_in_pix,
                "azim_1": azim_1,
                "azim_2": azim_2,
                "pix_coord": pix_coord,
                "x_geog": point_x,
                "y_geog": point_y,
            }

            # ---- Set up the analysis window ----
            dem.set_buffer(0)  # SINGLE mode: one point per output

            # Create the empty output raster file first
            # (this sets dem.gdal_output, needed by add_to_buffer)
            dem.write_output(output_path, compression=False)

            dem.set_master_window(
                max_radius,
                size_factor=precision,
                background_value=0,
                pad=precision > 0,
                curvature=useEarthCurvature,
                refraction=refraction)

            # ---- Run viewshed ----
            matrix_vis = ws.viewshed_raster(
                analysis_type, pt_dict, dem, interpolate=precision > 0)

            # ---- Apply mask ----
            mask_args = [radius_pix]
            mask_args.append(radius_in_pix)
            mask_args.append(azim_1)
            mask_args.append(azim_2)
            dem.set_mask(*mask_args)

            dem.add_to_buffer(matrix_vis, report=True)

            # ---- Flush and close output ----
            dem = None

            feedback.pushInfo(f"  Written: {output_path}")
            report.append(point_id)

            feedback.setProgress(int(((idx + 1) / total) * 100))

        # ---- Summary ----
        elapsed = round((time.process_time() - start) / 60, 2)
        txt = (f"\n Batch viewshed complete."
               f"\n Analysis time: {elapsed} minutes."
               f"\n {len(report)} viewshed(s) created in: {output_dir}")

        for pid in report:
            txt += f"\n  - viewshed_{pid}.tif"

        QgsMessageLog.logMessage(txt, "Viewshed info")
        feedback.pushInfo(txt)

        return {self.OUTPUT_DIR: output_dir}

    def name(self):
        return 'viewshed_batch'

    def displayName(self):
        return self.tr('Batch Viewshed from AOI')

    def group(self):
        return self.tr(self.groupId())

    def groupId(self):
        return 'Analysis'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return type(self)()
