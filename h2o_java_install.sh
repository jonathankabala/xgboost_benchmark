

# downlaod java
wget https://github.com/adoptium/temurin11-binaries/releases/download/jdk-11.0.20.1%2B1/OpenJDK11U-jdk_x64_linux_hotspot_11.0.20.1_1.tar.gz
mkdir -p $HOME/java

tar -xvzf OpenJDK11U-jdk_x64_linux_hotspot_11.0.20.1_1.tar.gz -C $HOME/java --strip-components=1


export PATH=$HOME/java/bin:$PATH

java --version


wget https://h2o-release.s3.amazonaws.com/h2o/rel-3.46.0/7/h2o-3.46.0.7.zip


unzip h2o-3.46.0.7.zip
cd h2o-3.46.0.7
java -jar h2o.jar

