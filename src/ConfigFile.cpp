#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <fstream>

void exitWithError(const std::string &error) 
{
  std::cout << error;
  std::cin.ignore();
  std::cin.get();

  exit(EXIT_FAILURE);
}
  
class Convert
{
public:
  template <typename T>
  static std::string T_to_string(T const &val) 
  {
    std::ostringstream ostr;
    ostr << val;

    return ostr.str();
  }
    
  template <typename T>
  static T string_to_T(std::string const &val) 
  {
    std::istringstream istr(val);
    T returnVal;
    if (!(istr >> returnVal))
      exitWithError("CFG: Not a valid " + (std::string)typeid(T).name() + " received!\n");

    return returnVal;
  } 

};

class ConfigFile
{
private:
  std::map<std::string, std::string> contents;
  std::string fName;
  std::vector<std::vector<float> > vertex;
  std::vector<std::vector<float> > normals;
  std::vector<std::vector<int > > index;
  
  int vertex_type=4;
  
  bool extract_normals = false;
  bool extract_colors = false;
  
  void removeComment(std::string &line) const
  {
    if (line.find(';') != line.npos)
      line.erase(line.find(';'));
  }

  bool onlyWhitespace(const std::string &line) const
  {
    return (line.find_first_not_of(' ') == line.npos);
  }
  bool validLine(const std::string &line) const
  {
    std::string temp = line;
    temp.erase(0, temp.find_first_not_of("\t "));
    if (temp[0] == '=')
      return false;

    for (size_t i = temp.find('=') + 1; i < temp.length(); i++)
      if (temp[i] != ' ')
        return true;

    return false;
  }

  void extractKey(std::string &key, size_t const &sepPos, const std::string &line) const
  {
    key = line.substr(0, sepPos);
    if (key.find('\t') != line.npos || key.find(' ') != line.npos)
      key.erase(key.find_first_of("\t "));
  }
  void extractValue(std::string &value, size_t const &sepPos, const std::string &line) const
  {
    value = line.substr(sepPos + 1);
    value.erase(0, value.find_first_not_of("\t "));
    value.erase(value.find_last_not_of("\t ") + 1);
  }

  void extractContents(const std::string &line) 
  {
    std::string temp = line;
    temp.erase(0, temp.find_first_not_of("\t "));
    size_t sepPos = temp.find('=');

    std::string key, value;
    extractKey(key, sepPos, temp);
    extractValue(value, sepPos, temp);  

    if (!keyExists(key))
      contents.insert(std::pair<std::string, std::string>(key, value));
    else
      exitWithError("CFG: Can only have unique key names!\n");
  }
  
  void extractVertex(const std::string &line)
  {
    std::string temp = line;
    std::vector<float> vertex_temp(3);
    std::vector<float> normals_temp(3);
    std::vector<int> index_temp(vertex_type);
    float temp_n;
    size_t first_pos = temp.find(' ');
    
    std::stringstream ss; 
		ss << temp.substr(0, first_pos);
		ss >> temp_n;
    
    //For Extracting face index information
    if (temp_n==vertex_type){
      temp.erase(0, first_pos+1);
      for (int i=0 ; i<vertex_type; i++){
        size_t pos = temp.find(' ');
        
        std::stringstream ss1; 
				ss1 << temp.substr(0, pos);
				ss1 >> index_temp[i];
		
        temp.erase(0, pos+1);
      }
      index.push_back(index_temp);
    }

    //For Extracting vertex information
    else if (extract_normals){
      
      for (int i=0 ; i<3; i++){
        size_t pos = temp.find(' ');
        
        std::stringstream ss1; 
				ss1 << temp.substr(0, pos);
				ss1 >> vertex_temp[i];

        temp.erase(0, pos+1);
      }
      
      for (int i=0 ; i<3; i++){
        size_t pos;
        if (i==2)
          pos = temp.find('\n');
        else
          pos = temp.find(' ');
          
        std::stringstream ss1; 
				ss1 << temp.substr(0, pos);
				ss1 >> normals_temp[i];
				
        temp.erase(0, pos+1);
      }
      normals.push_back(normals_temp);
      vertex.push_back(vertex_temp);
    }
    //~ else if (!extract_normals){
      //~ for (int i=0 ; i<3; i++){
        //~ size_t pos;
        //~ if (i==2)
          //~ pos = temp.find('\n');
        //~ else
          //~ pos = temp.find(' ');
        //~ vertex_temp[i] = std::stof(temp.substr(0, pos));
        //~ temp.erase(0, pos+1);
      //~ }
      //~ vertex.push_back(vertex_temp);
    //~ }
  }

  void parseLine(const std::string &line, size_t const lineNo)
  {
    if (line.find('=') == line.npos)
      //~ exitWithError("CFG: Couldn't find separator on line: " + Convert::T_to_string(lineNo) + "\n");
      extractVertex(line);

    if (!validLine(line))
      exitWithError("CFG: Bad format for line: " + Convert::T_to_string(lineNo) + "\n");

    //~ extractContents(line);
  }

  void ExtractKeys()
  {
    std::ifstream file;
    file.open(fName.c_str());
    if (!file)
      exitWithError("CFG: File " + fName + " couldn't be found!\n");

    std::string line;
    size_t lineNo = 0;
    while (std::getline(file, line))
    {
      lineNo++;
      std::string temp = line;
      //~ std::cout << "(Line number: " << lineNo << ") " << temp << std::endl;

      if (temp.empty())
        continue;

      removeComment(temp);
      if (onlyWhitespace(temp))
        continue;

      parseLine(temp, lineNo);
    }

    file.close();
  }
public:
  ConfigFile(const std::string &fName)
  {
    this->fName = fName;
  }

  bool keyExists(const std::string &key) const
  {
    return contents.find(key) != contents.end();
  }

  template <typename ValueType>
  ValueType getValueOfKey(const std::string &key, ValueType const &defaultValue = ValueType()) const
  {
    if (!keyExists(key))
      return defaultValue;

    return Convert::string_to_T<ValueType>(contents.find(key)->second);
  }
  
  void getVertex(std::vector<std::vector<float> >& retrieve ){
    retrieve.swap(vertex);
  }
  
  void getNormals(std::vector<std::vector<float> >& retrieve ){
    retrieve.swap(normals);
  }
  
  void getIndex(std::vector<std::vector<int> >& retrieve ){
    retrieve.swap(index);
  }
  
  void setVertexType(std::string s){
    if(s=="TRIANGLES"){
      vertex_type = 3;
    }else if(s == "QUADS"){
      vertex_type = 4;
    }
  }
  
  void toggleExtractNormals(bool t){
    extract_normals = t;
  }
  
  void toggleExtractColors(bool t){
    extract_colors = t;
  }
  
  void extractKeys(){
    ExtractKeys();
  }
  
};
