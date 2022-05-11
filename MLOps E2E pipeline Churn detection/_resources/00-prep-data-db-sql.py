# Databricks notebook source
# MAGIC %sql
# MAGIC create database if not exists field_demos_retail;
# MAGIC use field_demos_retail;

# COMMAND ----------

# MAGIC %sql 
# MAGIC create table if not exists field_demos_retail.customer_churn_bronze location '/mnt/field-demos/retail/customer_churn';
# MAGIC create table if not exists field_demos_retail.customer_churn_features location '/mnt/field-demos/retail/customer_churn_features';
# MAGIC create table if not exists field_demos_retail.customer_churn_predictions location '/mnt/field-demos/retail/customer_churn_predictions';

# COMMAND ----------

# MAGIC %sql 
# MAGIC show tables;