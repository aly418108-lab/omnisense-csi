# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ali7fa/esp/esp-idf/components/bootloader/subproject"
  "/home/ali7fa/.gemini/antigravity/scratch/omnisense-csi/firmware/esp32_tx_injector/build/bootloader"
  "/home/ali7fa/.gemini/antigravity/scratch/omnisense-csi/firmware/esp32_tx_injector/build/bootloader-prefix"
  "/home/ali7fa/.gemini/antigravity/scratch/omnisense-csi/firmware/esp32_tx_injector/build/bootloader-prefix/tmp"
  "/home/ali7fa/.gemini/antigravity/scratch/omnisense-csi/firmware/esp32_tx_injector/build/bootloader-prefix/src/bootloader-stamp"
  "/home/ali7fa/.gemini/antigravity/scratch/omnisense-csi/firmware/esp32_tx_injector/build/bootloader-prefix/src"
  "/home/ali7fa/.gemini/antigravity/scratch/omnisense-csi/firmware/esp32_tx_injector/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ali7fa/.gemini/antigravity/scratch/omnisense-csi/firmware/esp32_tx_injector/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ali7fa/.gemini/antigravity/scratch/omnisense-csi/firmware/esp32_tx_injector/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
