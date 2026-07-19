# Contratos de datos intermedios

## Objetivo

Este documento define la estructura, las reglas de preparación y el uso
previsto de los datasets intermedios del proyecto.

Los archivos intermedios se generan de forma reproducible a partir de los
datos originales y no se almacenan en Git.

## Reproducción

```terminal de cmd
python scripts/build_retail_interim.py
python scripts/build_crm_interim.py