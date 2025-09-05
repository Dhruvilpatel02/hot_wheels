# mongodb_database.py - MongoDB Integration for Hot Wheels Analysis System

import os
import json
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
import pymongo
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure
import gridfs
import io
import re
import logging
import csv
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HotWheelsMongoDatabase:
    """MongoDB database layer for Hot Wheels analysis caching and storage"""
    
    def __init__(self, connection_string: str = None, database_name: str = "cars"):
        self.connection_string = connection_string or "mongodb://localhost:27017/"
        self.database_name = database_name
        self.client = None
        self.db = None
        self.fs = None
        self._connect()
    
    def _connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            self.fs = gridfs.GridFS(self.db)
            logger.info(f"Connected to MongoDB: {self.database_name}")
            self._create_indexes()
        except ServerSelectionTimeoutError:
            logger.error("Failed to connect to MongoDB. Is it running?")
            self.client = None # Ensure client is None on failure
        except Exception as e:
            logger.error(f"MongoDB connection error: {e}")
            self.client = None

    def is_connected(self) -> bool:
        return self.client is not None

    def _create_indexes(self):
        """Create database indexes for optimal performance"""
        try:
            # Main collection for analysis results
            collection = self.db.analyses
            collection.create_index("storage_timestamp")
            collection.create_index("image_hash", unique=True)
            collection.create_index("generated_folder_name")
            # Text index for flexible searching
            collection.create_index([
                ("cars_analysis.quick_identification.Casting Name", "text"),
                ("cars_analysis.quick_identification.Real Car", "text"),
                ("cars_analysis.quick_identification.Series", "text")
            ])
            logger.info("Database indexes ensured.")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    def _calculate_image_hash(self, image_array: np.ndarray) -> str:
        """Calculate SHA-256 hash of an image for caching"""
        pil_img = Image.fromarray(image_array)
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format='JPEG', quality=95)
        return hashlib.sha256(img_bytes.getvalue()).hexdigest()

    def check_analysis_cache(self, original_image: np.ndarray) -> Optional[Dict]:
        """Check if analysis results exist for this image hash"""
        if not self.is_connected(): return None
        try:
            image_hash = self._calculate_image_hash(original_image)
            logger.info(f"Checking cache for image hash: {image_hash[:16]}...")
            cached_result = self.db.analyses.find_one({"image_hash": image_hash})
            
            if cached_result:
                logger.info(f"Cache HIT! Found analysis from {cached_result['storage_timestamp']}")
                # Convert ObjectIDs to strings for Streamlit compatibility
                cached_result['_id'] = str(cached_result['_id'])
                return cached_result
            else:
                logger.info("Cache MISS - Analysis not found.")
                return None
        except Exception as e:
            logger.error(f"Error checking analysis cache: {e}")
            return None
    
    def store_analysis_results(self, original_image: np.ndarray, crops: List[np.ndarray], analysis_results: List[Dict]) -> Tuple[bool, str]:
        """Store complete analysis results and generate a descriptive folder name"""
        if not self.is_connected(): return False, "db_error"
        try:
            image_hash = self._calculate_image_hash(original_image)
            timestamp = datetime.now()
            
            # Check if this hash already exists to prevent duplicates
            if self.db.analyses.find_one({"image_hash": image_hash}):
                logger.warning(f"Analysis with hash {image_hash[:16]} already exists. Skipping storage.")
                return True, "Already Exists"

            folder_name = generate_folder_name(analysis_results, timestamp)
            
            document = {
                'image_hash': image_hash,
                'storage_timestamp': timestamp.isoformat(),
                'generated_folder_name': folder_name,
                'cars_analysis': analysis_results,
                'total_cars': len(analysis_results),
            }
            
            self.db.analyses.insert_one(document)
            logger.info(f"Successfully stored analysis for {len(crops)} cars as '{folder_name}'")
            return True, folder_name
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")
            return False, f"storage_error_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    def search_by_car_name(self, car_name: str, limit: int = 20) -> List[Dict]:
        """Search analyses using the text index for a car name."""
        if not self.is_connected(): return []
        try:
            # Using text search for better matching across multiple fields
            results = list(self.db.analyses.find(
                {'$text': {'$search': f'"{car_name}"'}},
                {'score': {'$meta': 'textScore'}}
            ).sort([('score', {'$meta': 'textScore'})]).limit(limit))
            return [self._serialize_doc(doc) for doc in results]
        except Exception as e:
            logger.error(f"Error searching by car name '{car_name}': {e}")
            return []

    def get_all_car_names(self) -> List[Dict]:
        """Get a list of all unique car names, their count, and last seen date."""
        if not self.is_connected(): return []
        try:
            pipeline = [
                {'$unwind': '$cars_analysis'},
                {'$match': {'cars_analysis.quick_identification.Casting Name': {'$ne': 'N/A', '$ne': 'Unidentified'}}},
                {'$group': {
                    '_id': '$cars_analysis.quick_identification.Casting Name',
                    'count': {'$sum': 1},
                    'last_seen': {'$max': '$storage_timestamp'}
                }},
                {'$sort': {'count': -1, 'last_seen': -1}},
                {'$limit': 50},
                {'$project': {'name': '$_id', 'count': 1, 'last_seen': 1, '_id': 0}}
            ]
            return list(self.db.analyses.aggregate(pipeline))
        except Exception as e:
            logger.error(f"Error aggregating car names: {e}")
            return []

    def get_database_statistics(self) -> Dict:
        """Get overall database statistics."""
        if not self.is_connected(): return {}
        try:
            total_analyses = self.db.analyses.count_documents({})
            
            # Using aggregation to get total cars correctly
            cars_agg = list(self.db.analyses.aggregate([
                {'$group': {'_id': None, 'total_cars': {'$sum': '$total_cars'}}}
            ]))
            total_cars_count = cars_agg[0]['total_cars'] if cars_agg else 0
            
            db_stats = self.db.command('dbStats')
            storage_size_mb = db_stats.get('storageSize', 0) / (1024 * 1024)

            return {
                'total_analyses': total_analyses,
                'total_cars_analyzed': total_cars_count,
                'storage_size_mb': storage_size_mb,
            }
        except Exception as e:
            logger.error(f"Error getting DB stats: {e}")
            return {}

    def delete_old_analyses(self, days_old: int) -> Dict:
        """Delete analyses older than a specified number of days."""
        if not self.is_connected(): return {'error': 'Not connected'}
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
            result = self.db.analyses.delete_many({'storage_timestamp': {'$lt': cutoff_date}})
            logger.info(f"Deleted {result.deleted_count} analyses older than {days_old} days.")
            return {'deleted_count': result.deleted_count}
        except Exception as e:
            logger.error(f"Error deleting old analyses: {e}")
            return {'error': str(e)}
            
    def _serialize_doc(self, doc):
        """Convert MongoDB document to a JSON-serializable dictionary."""
        if '_id' in doc:
            doc['_id'] = str(doc['_id'])
        return doc
        
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# --- Helper & Utility Functions ---

def generate_folder_name(analysis_results: List[Dict], timestamp: datetime) -> str:
    """Generates a descriptive folder name from analysis results."""
    car_names = []
    for r in analysis_results:
        name = r.get('quick_identification', {}).get('Casting Name', 'unidentified')
        if name not in ['N/A', 'unidentified', 'Unidentified']:
            car_names.append(name)
    
    # Clean up names for filesystem
    car_names = [re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_') for name in car_names]
    car_names = [name for name in car_names if name] # Remove empty names

    time_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    if not car_names:
        return f"unidentified_cars_{len(analysis_results)}x_{time_str}"
    
    if len(car_names) == 1:
        return f"{car_names[0]}_{time_str}"
    
    if len(car_names) <= 3:
        return f"{'_and_'.join(car_names)}_{time_str}"
        
    return f"{car_names[0]}_plus_{len(car_names)-1}_more_{time_str}"

# --- Public API Functions for Streamlit App ---

def enhanced_analysis_with_database_check(api_key: str, original_image: np.ndarray, crops: List[np.ndarray], db_connection_string: str = None) -> Tuple[List[Dict], bool, str]:
    """
    Checks database first, then falls back to fresh analysis, and stores the result.
    Returns: (analysis_results, from_cache, final_folder_name)
    """
    from methods import process_all_cars_enhanced # Local import to avoid circular dependency issues
    
    db = HotWheelsMongoDatabase(db_connection_string)
    
    try:
        if not db.is_connected():
            st.error("Database connection failed. Proceeding with fresh analysis without caching.")
            analysis_results = process_all_cars_enhanced(api_key, crops)
            return analysis_results, False, "db_connection_error"

        cached_result = db.check_analysis_cache(original_image)
        
        if cached_result:
            return cached_result['cars_analysis'], True, cached_result['generated_folder_name']
        
        # Perform fresh analysis
        analysis_results = process_all_cars_enhanced(api_key, crops)
        
        # Store results
        success, folder_name = db.store_analysis_results(original_image, crops, analysis_results)
        if not success:
            logger.warning("Failed to store analysis results in the database.")
        
        return analysis_results, False, folder_name
    finally:
        db.close_connection()

def get_database_dashboard_info(db_connection_string: str = None) -> Dict:
    """Get database information for the dashboard display."""
    db = HotWheelsMongoDatabase(db_connection_string)
    try:
        if not db.is_connected():
            return {'connection_status': 'Connection Failed', 'statistics': {}, 'recent_analyses': []}
            
        stats = db.get_database_statistics()
        return {'connection_status': 'Connected', 'statistics': stats}
    finally:
        db.close_connection()

def search_cars_by_name(car_name: str, db_connection_string: str = None) -> List[Dict]:
    db = HotWheelsMongoDatabase(db_connection_string)
    try:
        return db.search_by_car_name(car_name)
    finally:
        db.close_connection()

def get_all_car_names_list(db_connection_string: str = None) -> List[Dict]:
    db = HotWheelsMongoDatabase(db_connection_string)
    try:
        return db.get_all_car_names()
    finally:
        db.close_connection()

def browse_cars_by_category(category: str, db_connection_string: str = None, limit: int = 20) -> List[Dict]:
    db = HotWheelsMongoDatabase(db_connection_string)
    try:
        query = {}
        sort_key = [('storage_timestamp', -1)]
        
        if category == 'high_confidence':
            query = {'cars_analysis.confidence_level': 'High'}
        elif category == 'treasure_hunts':
            query = {'cars_analysis.comprehensive_report': {'$regex': 'treasure hunt', '$options': 'i'}}
        elif category == 'premium':
            query = {'cars_analysis.comprehensive_report': {'$regex': 'premium', '$options': 'i'}}
        elif category == 'popular':
            # This is handled differently; get_all_car_names_list is better for this UI feature
            return [] # Logic is in the UI for this one now.

        results = list(db.db.analyses.find(query).sort(sort_key).limit(limit))
        return [db._serialize_doc(doc) for doc in results]
    finally:
        db.close_connection()

def get_car_statistics(db_connection_string: str = None) -> Dict:
    db = HotWheelsMongoDatabase(db_connection_string)
    try:
        if not db.is_connected(): return {}
        
        stats = {}
        # Basic counts
        total_analyses = db.db.analyses.count_documents({})
        cars_agg = list(db.db.analyses.aggregate([{'$group': {'_id': None, 'total_cars': {'$sum': '$total_cars'}}}]))
        stats['basic'] = {'total_analyses': total_analyses, 'total_cars': cars_agg[0]['total_cars'] if cars_agg else 0}
        
        # Confidence distribution
        conf_pipeline = [{'$unwind': '$cars_analysis'}, {'$group': {'_id': '$cars_analysis.confidence_level', 'count': {'$sum': 1}}}]
        conf_data = list(db.db.analyses.aggregate(conf_pipeline))
        stats['confidence'] = {item['_id'].lower()+'_confidence': item['count'] for item in conf_data if item['_id']}

        # Most common cars
        stats['most_common_cars'] = db.get_all_car_names()[:10]
        
        # Series distribution
        series_pipeline = [
            {'$unwind': '$cars_analysis'},
            {'$match': {'cars_analysis.quick_identification.Series': {'$ne': 'N/A', '$ne': None}}},
            {'$group': {'_id': '$cars_analysis.quick_identification.Series', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}, {'$limit': 10}
        ]
        stats['series_distribution'] = list(db.db.analyses.aggregate(series_pipeline))
        return stats
    finally:
        db.close_connection()

def cleanup_database(db_connection_string: str = None, days_old: int = 90) -> Dict:
    db = HotWheelsMongoDatabase(db_connection_string)
    try:
        return db.delete_old_analyses(days_old)
    finally:
        db.close_connection()

def export_car_collection(db_connection_string: str = None, export_format: str = 'json') -> str:
    db = HotWheelsMongoDatabase(db_connection_string)
    try:
        if not db.is_connected(): return "Error: Could not connect to database."
        
        all_cars = []
        for analysis in db.db.analyses.find({}):
            for car in analysis.get('cars_analysis', []):
                quick_info = car.get('quick_identification', {})
                all_cars.append({
                    'analysis_date': analysis.get('storage_timestamp', ''),
                    'car_number': car.get('crop_number', 0),
                    'confidence': car.get('confidence_level', ''),
                    'casting_name': quick_info.get('Casting Name', ''),
                    'real_car': quick_info.get('Real Car', ''),
                    'color': quick_info.get('Color/Finish', ''),
                    'series': quick_info.get('Series', ''),
                    'first_released': quick_info.get('First Released', ''),
                })
        
        if not all_cars: return "No cars to export."

        if export_format == 'csv':
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=all_cars[0].keys())
            writer.writeheader()
            writer.writerows(all_cars)
            return output.getvalue()
        
        return json.dumps(all_cars, indent=2, default=str)
    except Exception as e:
        return f"Export error: {e}"
    finally:
        db.close_connection()