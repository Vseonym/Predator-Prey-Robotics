cmake_minimum_required(VERSION 3.5)
project(thymio_description)

find_package(ament_cmake REQUIRED)
find_package(urdf REQUIRED)

install(DIRECTORY thymio/meshes/
  DESTINATION share/${PROJECT_NAME}/meshes
)

install(DIRECTORY thymio/urdf launch
  DESTINATION share/${PROJECT_NAME}
)

install(FILES thymio/model.config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()