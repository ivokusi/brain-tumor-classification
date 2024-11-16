#!/bin/bash

az login

read -p "Create resource group? (y/n): " create_resource_group

if [ "$create_resource_group" == "n" ]; then
    az group list --output table
fi

read -p "Enter the resoure group name: " resource_group
read -p "Enter the location: " location

if [ "$create_resource_group" == "y" ]; then
    az group create --name $resource_group --location $location
fi

read -p "Enter the app service plan name: " app_service_plan

az appservice plan create --name $app_service_plan --resource-group $resource_group --sku B1 --is-linux

read -p "Enter the app service name: " app_service

az webapp create --resource-group $resource_group --plan $app_service_plan --name $app_service --runtime "PYTHON:3.12"

az webapp list --resource-group $resource_group --output table

read -p "Enter .env file path: " env_file

# Read .env file line by line
while IFS= read -r line || [ -n "$line" ]; do
    # Ignore empty lines and comments
    if [[ -z "$line" || "$line" == \#* ]]; then
        continue
    fi

    # Extract key and value
    key=$(echo "$line" | cut -d '=' -f 1)
    value=$(echo "$line" | cut -d '=' -f 2-)

    # Add to Azure App Service
    echo "Adding $key to Azure App Service..."
    az webapp config appsettings set --resource-group "$resource_group" --name "$app_service" --settings "$key=$value"

done < "$env_file"

read -p "Enter startup file path: " startup_file

az webapp config set --resource-group $resource_group --name $app_service --startup-file $startup_file

az webapp restart --resource-group $resource_group --name $app_service
