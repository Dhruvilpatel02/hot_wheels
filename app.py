# updated_app.py - Streamlit App with MongoDB Database Integration

import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import os
import shutil

# Import your existing modules
import main as detection_engine
from methods import (
    api_manager,
    parse_bbox_json,
    draw_bboxes_rgb,
    crop_and_upscale,
    extract_quick_display_info
)

# Import the new MongoDB database functions
from mongodb_database import (
    enhanced_analysis_with_database_check,
    search_cars_by_name,
    get_all_car_names_list,
    get_database_dashboard_info,
    browse_cars_by_category,
    get_car_statistics,
    cleanup_database,
    export_car_collection
)

def display_database_status():
    """Display MongoDB database connection and statistics"""
    st.subheader("📊 Database Status")
    
    db_info = get_database_dashboard_info(st.session_state.get('mongodb_connection'))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if db_info['connection_status'] == 'Connected':
            st.success("🟢 MongoDB Connected")
        else:
            st.error(f"🔴 {db_info['connection_status']}")
    
    with col2:
        total_analyses = db_info.get('statistics', {}).get('total_analyses', 0)
        st.metric("Total Analyses", total_analyses)
    
    with col3:
        total_cars = db_info.get('statistics', {}).get('total_cars_analyzed', 0)
        st.metric("Cars Analyzed", total_cars)
    
    with col4:
        storage_size = db_info.get('statistics', {}).get('storage_size_mb', 0)
        st.metric("Storage Used", f"{storage_size:.1f} MB")

def display_car_search_interface():
    """Enhanced car search interface"""
    st.subheader("🔍 Search Your Car Collection")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Name Search", "📂 Browse Categories", "📈 Popular Cars"])
    
    with tab1:
        search_term = st.text_input(
            "Search by car name:",
            placeholder="e.g., Lamborghini, Porsche 911, Custom Chevy",
            help="Search for cars by casting name, real car model, or series"
        )
        
        if search_term:
            with st.spinner("Searching database..."):
                results = search_cars_by_name(search_term, st.session_state.get('mongodb_connection'))
            
            if results:
                st.success(f"Found {len(results)} matching analyses")
                display_search_results(results)
            else:
                st.info("No cars found matching your search. Try different keywords.")
    
    with tab2:
        category = st.selectbox(
            "Browse by category:",
            ["recent", "high_confidence", "popular", "treasure_hunts", "premium"],
            format_func=lambda x: {
                "recent": "🕒 Recent Analyses",
                "high_confidence": "🏆 High Confidence IDs", 
                "popular": "⭐ Popular Cars",
                "treasure_hunts": "💎 Treasure Hunts",
                "premium": "✨ Premium Series"
            }.get(x, x.replace('_', ' ').title())
        )
        
        if st.button("Browse Category", type="primary"):
            with st.spinner(f"Loading {category.replace('_', ' ')} cars..."):
                results = browse_cars_by_category(category, st.session_state.get('mongodb_connection'))
            
            if results:
                st.success(f"Found {len(results)} results in '{category.replace('_', ' ')}' category")
                display_search_results(results)
            else:
                st.info(f"No cars found in the '{category.replace('_', ' ')}' category.")
    
    with tab3:
        st.markdown("**Most Common Cars in Your Collection:**")
        popular_cars = get_all_car_names_list(st.session_state.get('mongodb_connection'))
        
        if popular_cars:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1: st.markdown("**Car Name**")
            with col2: st.markdown("**Count**")
            with col3: st.markdown("**Last Seen**")

            for i, car_info in enumerate(popular_cars[:15], 1):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1: st.markdown(f"{i}. {car_info['name']}")
                with col2: st.markdown(f"**{car_info['count']}x**")
                with col3: st.markdown(f"`{car_info['last_seen'][:10]}`")
        else:
            st.info("No car data available to show popular models.")

def display_search_results(results):
    """Display search results in a consistent format"""
    for i, result in enumerate(results):
        folder_name = result.get('generated_folder_name', result.get('folder_name', 'Unknown'))
        timestamp = result.get('storage_timestamp', 'Unknown')
        car_count = len(result.get('cars_analysis', []))
        
        with st.expander(f"📁 {folder_name} ({car_count} cars) - {timestamp[:10]}", expanded=i < 3):
            for car in result.get('cars_analysis', []):
                st.markdown(f"**Car #{car.get('car_number', 'N/A')}** | Confidence: `{car.get('confidence_level', 'Unknown')}`")
                
                quick_info = car.get('quick_identification', {})
                if quick_info:
                    info_text = [f"**{key}**: {value}" for key, value in quick_info.items() if value and value not in ['N/A', 'Unknown', '']]
                    if info_text:
                        st.markdown(" | ".join(info_text))
                
                report = car.get('comprehensive_report', '')
                if report:
                    excerpt = report[:250] + "..." if len(report) > 250 else report
                    st.markdown(f"> *{excerpt}*")
                
                st.markdown("---")

def display_database_management():
    """Database management interface"""
    st.subheader("🛠️ Database Management")
    
    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "🧹 Maintenance", "📤 Export"])
    
    with tab1:
        st.markdown("### Detailed Collection Statistics")
        if st.button("🔄 Refresh Statistics"):
            with st.spinner("Calculating statistics..."):
                stats = get_car_statistics(st.session_state.get('mongodb_connection'))
            
            if stats:
                st.markdown("#### Overview")
                basic = stats.get('basic', {})
                col1, col2 = st.columns(2)
                with col1: st.metric("Total Analyses", basic.get('total_analyses', 0))
                with col2: st.metric("Total Unique Cars", basic.get('total_cars', 0))
                
                st.markdown("#### Identification Confidence")
                confidence = stats.get('confidence', {})
                if confidence:
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("High Confidence", confidence.get('high_confidence', 0))
                    with col2: st.metric("Medium Confidence", confidence.get('medium_confidence', 0))  
                    with col3: st.metric("Low Confidence", confidence.get('low_confidence', 0))
                
                common_cars = stats.get('most_common_cars', [])
                if common_cars:
                    st.markdown("#### Most Common Cars")
                    for car in common_cars[:10]: st.text(f"- {car['_id']}: {car['count']} times")
                
                series_dist = stats.get('series_distribution', [])
                if series_dist:
                    st.markdown("#### Series Distribution") 
                    for series in series_dist[:10]:
                        if series['_id']: st.text(f"- {series['_id']}: {series['count']} cars")
            else:
                st.error("Could not retrieve statistics.")

    with tab2:
        st.markdown("### Database Cleanup")
        st.warning("**Warning:** This will permanently delete data.", icon="⚠️")
        
        days_old = st.slider("Delete analyses older than (days):", 30, 365, 90)
        
        if st.button("🧹 Clean Old Data", type="secondary"):
            with st.spinner(f"Deleting analyses older than {days_old} days..."):
                cleanup_result = cleanup_database(st.session_state.get('mongodb_connection'), days_old)
            
            if cleanup_result and 'error' not in cleanup_result:
                st.success("Database cleanup completed!")
                st.json(cleanup_result)
            else:
                st.error(f"Database cleanup failed: {cleanup_result.get('error', 'Unknown error')}")
    
    with tab3:
        st.markdown("### Export Car Collection")
        export_format = st.selectbox("Export format:", ["json", "csv"])
        
        if st.button("📤 Export Collection", type="primary"):
            with st.spinner("Exporting data..."):
                exported_data = export_car_collection(st.session_state.get('mongodb_connection'), export_format)
            
            if exported_data and not exported_data.startswith("Export error"):
                st.success("Export completed!")
                filename = f"hotwheels_collection_{datetime.now().strftime('%Y%m%d')}.{export_format}"
                st.download_button(
                    label=f"💾 Download {export_format.upper()} File",
                    data=exported_data,
                    file_name=filename,
                    mime=f"application/{export_format}" if export_format == "json" else "text/csv"
                )
            else:
                st.error(f"Export failed: {str(exported_data)}")

def main_app_with_database():
    st.set_page_config(
        page_title="🏆 Hot Wheels AI Analyzer", 
        page_icon="🚗", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏆 Hot Wheels AI Analyzer")
    st.subheader("🔍 Database-Powered Car Identification & Collection Management")
    
    if 'mongodb_connection' not in st.session_state:
        st.session_state.mongodb_connection = "mongodb://localhost:27017/"
    
    with st.sidebar:
        st.header("🗄️ Database Settings")
        st.session_state.mongodb_connection = st.text_input(
            "MongoDB Connection String:",
            value=st.session_state.mongodb_connection,
            help="Default: mongodb://localhost:27017/ for local MongoDB"
        )
        st.markdown("---")
        st.header("🔧 Analysis Settings")
        padding = st.slider("Crop Padding", 0, 80, 10, 5)
        upscale_factor = st.slider("Upscale Factor", 1, 6, 3, 1)
        sharpen = st.checkbox("Apply Sharpening", value=True)
        st.markdown("---")
        st.header("📋 Display Options")
        show_database_status = st.checkbox("Show Database Status", value=True)
        show_detection = st.checkbox("Show Detection Results", value=True)
    
    main_tab1, main_tab2, main_tab3 = st.tabs(["🏠 Analysis", "🔍 Search & Manage", "📊 Collection Stats"])
    
    with main_tab1:
        if show_database_status:
            display_database_status()
            st.markdown("---")
        
        st.markdown("### 📤 Upload Hot Wheels Image")
        img_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear image of your Hot Wheels cars for analysis"
        )

        if img_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use a temporary directory for all processing files for this run
            run_dir = os.path.join("temp_runs", timestamp)
            os.makedirs(run_dir, exist_ok=True)
            
            original_img = Image.open(img_file).convert("RGB")
            original_img_array = np.array(original_img)
            
            st.image(original_img_array, caption="Uploaded Image", use_column_width=True)
            
            # <<< START: MODIFIED SECTION >>>
            # Save the uploaded file to a temporary path to pass to the detection engine
            temp_img_path = os.path.join(run_dir, img_file.name)
            with open(temp_img_path, "wb") as f:
                f.write(img_file.getbuffer())

            with st.spinner("Step 1/4: Detecting and validating cars..."):
                # Call the correct function from your main.py
                processing_result = detection_engine.run_detection_and_classification(
                    image_input=temp_img_path,
                    output_dir=run_dir
                )

            # Check the dictionary returned by your function
            if processing_result.get("status") != "accepted":
                reason = processing_result.get("reason", "An unknown error occurred during validation.")
                st.error(f"Image Rejected: {reason}")
                shutil.rmtree(run_dir) # Clean up temp directory
                st.stop()
            
            # Extract the correct JSON file path from the result dictionary
            detection_result_path = processing_result.get("locations_file")
            annotated_img_path = processing_result.get("annotated_path")
            
            if not detection_result_path or not os.path.exists(detection_result_path):
                st.error("Detection succeeded, but the location JSON file was not found.")
                shutil.rmtree(run_dir) # Clean up temp directory
                st.stop()
            
            detections = parse_bbox_json(detection_result_path)
            # <<< END: MODIFIED SECTION >>>

            if not detections:
                st.warning("No cars were detected in the uploaded image.")
                shutil.rmtree(run_dir) # Clean up temp directory
                st.stop()
            
            st.success(f"✅ Step 1 complete: Found and validated {len(detections)} cars.")

            if show_detection and annotated_img_path and os.path.exists(annotated_img_path):
                annotated_img = Image.open(annotated_img_path)
                st.image(annotated_img, caption="Detection Results", use_column_width=True)

            with st.spinner("Step 2/4: Cropping and preparing images..."):
                crops = crop_and_upscale(original_img_array, detections, padding, upscale_factor, sharpen)
            
            st.success("✅ Step 2 complete: Images prepared for analysis.")

            st.write("### 🚗 Detected Cars for Analysis")
            if len(crops) > 0:
                cols = st.columns(len(crops))
                for i, crop_img in enumerate(crops):
                    with cols[i]:
                        st.image(crop_img, caption=f"Car #{i+1}", use_column_width=True)
            
            st.markdown("---")
            st.info("Step 3/4: Starting AI analysis. This may take a minute...")

            analysis_results, from_cache, final_folder_name = enhanced_analysis_with_database_check(
                api_key=None,
                original_image=original_img_array,
                crops=crops,
                db_connection_string=st.session_state.mongodb_connection
            )
            
            if from_cache:
                st.success(f"✅ Analysis complete! Results were retrieved from the database cache.")
            else:
                st.success(f"✅ Analysis complete! New results saved to the database as '{final_folder_name}'.")

            st.balloons()
            st.header("📋 Final Analysis Report")
            
            for result in analysis_results:
                car_num = result.get('crop_number', 'N/A')
                with st.expander(f"### 🏎️ Car #{car_num} Report", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if car_num > 0 and (car_num - 1) < len(crops):
                            st.image(crops[car_num-1], caption=f"Car #{car_num}")
                    with col2:
                        quick_info = result.get('quick_identification', {})
                        for key, value in quick_info.items():
                            if value and value.lower() not in ["n/a", "unknown"]:
                                st.markdown(f"**{key}:** {value}")
                    
                    st.markdown("#### 📜 Comprehensive Report")
                    st.markdown(result.get('comprehensive_report', 'No report generated.'))
            
            # Clean up the temporary run directory
            try:
                shutil.rmtree(run_dir)
            except Exception as e:
                st.warning(f"Could not clean up temporary directory '{run_dir}': {e}")

    with main_tab2:
        display_car_search_interface()

    with main_tab3:
        display_database_management()

if __name__ == "__main__":
    main_app_with_database()