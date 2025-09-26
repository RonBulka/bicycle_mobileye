#!/bin/bash

# Startup Check Script for Bicycle Safety System
# This script ensures all required hardware is ready before starting the main application

set -e

# Configuration
LOG_FILE="/tmp/bicycle_startup.log"
MAX_WAIT_TIME=60  # Maximum wait time in seconds
CAMERA_VENDOR_ID="03e7"  # OAK-D camera vendor ID

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to check if USB camera is connected
check_usb_camera() {
    log_message "Checking for OAK-D camera..."
    
    local wait_time=0
    while [ $wait_time -lt $MAX_WAIT_TIME ]; do
        if lsusb | grep -q "$CAMERA_VENDOR_ID"; then
            log_message "✓ OAK-D camera detected"
            return 0
        fi
        
        log_message "Waiting for camera... ($wait_time/$MAX_WAIT_TIME seconds)"
        sleep 2
        wait_time=$((wait_time + 2))
    done
    
    log_message "✗ Camera not detected after $MAX_WAIT_TIME seconds"
    return 1
}

# Function to check if Bluetooth service is running
check_bluetooth_service() {
    log_message "Checking Bluetooth service..."
    
    local wait_time=0
    while [ $wait_time -lt $MAX_WAIT_TIME ]; do
        if systemctl is-active --quiet bluetooth; then
            log_message "✓ Bluetooth service is active"
            return 0
        fi
        
        log_message "Waiting for Bluetooth service... ($wait_time/$MAX_WAIT_TIME seconds)"
        sleep 2
        wait_time=$((wait_time + 2))
    done
    
    log_message "✗ Bluetooth service not active after $MAX_WAIT_TIME seconds"
    return 1
}

# Function to connect to Bluetooth speaker
connect_bluetooth_speaker() {
    log_message "Attempting to connect to Bluetooth speaker..."
    
    # Wait a bit for Bluetooth to fully initialize
    sleep 3
    
    # Check if we have any saved Bluetooth devices
    local saved_devices=$(bluetoothctl devices | grep -E "(speaker|Speaker|SPEAKER|headphone|Headphone|HEADPHONE|Jabra|jabra)" || true)
    
    if [ -z "$saved_devices" ]; then
        log_message "No saved Bluetooth speakers found"
        return 0
    fi
    
    # Try to connect to the first available speaker
    local device_mac=$(echo "$saved_devices" | head -1 | awk '{print $2}')
    local device_name=$(echo "$saved_devices" | head -1 | awk '{for(i=3;i<=NF;i++) printf "%s ", $i; print ""}' | sed 's/ $//')
    
    log_message "Found device: $device_name ($device_mac)"
    
    # Try to connect
    if bluetoothctl connect "$device_mac" 2>/dev/null; then
        log_message "✓ Successfully connected to $device_name"
        return 0
    else
        log_message "⚠ Failed to connect to $device_name, continuing anyway"
        return 0
    fi
}

# Function to check audio system
check_audio_system() {
    log_message "Checking audio system..."
    
    # Check if PulseAudio is running
    if pgrep -x "pulseaudio" > /dev/null; then
        log_message "✓ PulseAudio is running"
    else
        log_message "⚠ PulseAudio not running, starting it..."
        pulseaudio --start --log-level=0 --log-target=syslog &
        sleep 2
    fi
    
    # Check if we have audio devices
    if pactl list short sinks | grep -q .; then
        log_message "✓ Audio sinks available"
        pactl list short sinks | while read -r sink_id sink_name sink_description; do
            log_message "  - Sink $sink_id: $sink_name"
        done
    else
        log_message "⚠ No audio sinks available"
    fi
}

# Function to check USB permissions
check_usb_permissions() {
    log_message "Checking USB permissions..."
    
    # Check if user is in plugdev group
    if groups | grep -q plugdev; then
        log_message "✓ User is in plugdev group"
    else
        log_message "⚠ User not in plugdev group, USB access may be limited"
    fi
}

# Function to check system resources
check_system_resources() {
    log_message "Checking system resources..."
    
    # Check available disk space (need at least 1GB)
    local available_space=$(df / | awk 'NR==2 {print $4}')
    if [ "$available_space" -gt 1048576 ]; then
        log_message "✓ Sufficient disk space available"
    else
        log_message "⚠ Low disk space: $(($available_space / 1024))MB available"
    fi
    
    # Check memory
    local total_mem=$(free -m | awk 'NR==2{print $2}')
    local available_mem=$(free -m | awk 'NR==2{print $7}')
    log_message "Memory: ${available_mem}MB available out of ${total_mem}MB total"
    
    # Check CPU temperature (if available)
    if [ -f "/sys/class/thermal/thermal_zone0/temp" ]; then
        local temp=$(($(cat /sys/class/thermal/thermal_zone0/temp) / 1000))
        log_message "CPU temperature: ${temp}°C"
        
        if [ "$temp" -gt 70 ]; then
            log_message "⚠ High CPU temperature: ${temp}°C"
        fi
    fi
}

# Function to check project directories
check_project_directories() {
    log_message "Checking project directories..."
    
    # Get the directory where this script is located
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_dir="$(dirname "$script_dir")"
    
    log_message "Project directory: $project_dir"
    
    # Check vid_result directory
    local vid_result_dir="$project_dir/vid_result"
    if [ -d "$vid_result_dir" ]; then
        log_message "✓ vid_result directory exists"
        
        # Check if it's writable
        if [ -w "$vid_result_dir" ]; then
            log_message "✓ vid_result directory is writable"
        else
            log_message "⚠ vid_result directory is not writable"
            return 1
        fi
    else
        log_message "Creating vid_result directory..."
        if mkdir -p "$vid_result_dir"; then
            log_message "✓ Created vid_result directory"
        else
            log_message "✗ Failed to create vid_result directory"
            return 1
        fi
    fi
    
    # Check log directory
    local log_dir="$project_dir/logs"
    if [ -d "$log_dir" ]; then
        log_message "✓ log directory exists"
        
        # Check if it's writable
        if [ -w "$log_dir" ]; then
            log_message "✓ log directory is writable"
        else
            log_message "⚠ log directory is not writable"
            return 1
        fi
    else
        log_message "Creating log directory..."
        if mkdir -p "$log_dir"; then
            log_message "✓ Created log directory"
        else
            log_message "✗ Failed to create log directory"
            return 1
        fi
    fi
    
    # Check available space in project directory
    local project_space=$(df "$project_dir" | awk 'NR==2 {print $4}')
    if [ "$project_space" -gt 524288 ]; then  # At least 512MB
        log_message "✓ Sufficient space in project directory: $(($project_space / 1024))MB available"
    else
        log_message "⚠ Low space in project directory: $(($project_space / 1024))MB available"
    fi
    
    return 0
}

# Main execution
main() {
    # Create log file
    touch "$LOG_FILE"

    log_message "=== Bicycle Mobileye Startup Check ==="
    
    # Check system resources first
    check_system_resources
    
    # Check project directories
    if ! check_project_directories; then
        log_message "ERROR: Project directory check failed, exiting"
        exit 1
    fi
    
    # Check USB permissions
    check_usb_permissions
    
    # Check and wait for USB camera
    if ! check_usb_camera; then
        log_message "ERROR: Camera not available, exiting"
        exit 1
    fi
    
    # Check and wait for Bluetooth service
    if ! check_bluetooth_service; then
        log_message "ERROR: Bluetooth service not available, exiting"
        exit 1
    fi
    
    # Check audio system
    check_audio_system
    
    # Try to connect to Bluetooth speaker
    connect_bluetooth_speaker
    
    log_message "=== All checks passed, system ready ==="
    exit 0
}

# Run main function
main "$@"
