@echo off
setlocal enabledelayedexpansion
REM Run all loss presets sequentially (Windows)
REM Usage: run_all_presets.bat

cd /d "%~dp0"

echo === Running loss presets ===
echo.

for %%P in (l1 charbonnier l2 ms_ssim log_l1 gradient ffl tv l1_ssim_baseline char_msssim char_msssim_grad char_msssim_grad_ffl char_msssim_logl1 char_msssim_logl1_ffl full_stack_tv geo_char_msssim geo_char_msssim_plus_ffl geo_char_msssim_grad uncert_char_msssim_grad uncert_full_stack uncert_char_msssim_logl1) do (
  for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "DT=%%I"
  set "TIMESTAMP=!DT:~0,4!!DT:~4,2!!DT:~6,2!_!DT:~8,2!!DT:~10,2!!DT:~12,2!"
  set "RUN_NAME=!TIMESTAMP!_%%P"
  echo --- [!TIMESTAMP!] Starting: %%P ---
  python src\train.py --loss-preset "%%P" --run-name "!RUN_NAME!"
  echo --- [%%P] Done ---
  echo.
)

echo === All presets complete ===
