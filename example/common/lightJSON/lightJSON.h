/*
MIT License

Copyright (c) 2020 Zhixuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef LIGHTJSON_H
#define LIGHTJSON_H

#include <algorithm>
#include <cassert>
#include <exception>
#include <iostream>
#include <type_traits>
// containers
#include <stack>
#include <vector>
// helpers for the parser
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>

// Macrop options
#ifndef LIGHTJSON_RT_TYPE_CHECK
#define LIGHTJSON_RT_TYPE_CHECK 1
#endif  // LIGHTJSON_RT_TYPE_CHECK

#ifndef LIGHTJSON_RT_BOUND_CHECK
#define LIGHTJSON_RT_BOUND_CHECK 1
#endif

#ifndef LIGHTJSON_INTEROPERABILITY
#define LIGHTJSON_INTEROPERABILITY 0
#endif  // LIGHTJSON_INTEROPERABILITY

#if LIGHTJSON_INTEROPERABILITY
#include "Core/Vec.h"
#endif

/*
TODO
 - [x] Optimize the allocation of jsNode.
 - [ ] Add support for comments.
 - [ ] Improve the speed of parsing strings.
 - [ ] Perform a complete profile.
 - [ ] Provide an iterator interface.
 - [X] Provide interoperability with SciCore
*/

namespace lightJSON {

// forward declarations
class jsonNode;
class jsonParser;

// implementations of the data types
using jsBool = bool;
using jsNumber = double;
using jsString = std::string;
using jsArray = std::vector<const jsonNode *>;
using jsMember = std::pair<std::string, const jsonNode *>;
using jsObject = std::vector<jsMember>;

enum NodeType { Null = 0, Bool, Number, String, Array, Object };

/**
 * Represent a node in the JSON tree.
 *
 * A node can be associated with different types of data,
 * e.g. a literal or a container.
 */
class jsonNode {
protected:
  enum {
    PadSize = std::max({sizeof(jsBool),
                        sizeof(jsNumber),
                        sizeof(jsString),
                        sizeof(jsArray),
                        sizeof(jsObject)})
  };
  std::size_t nodeType;
  unsigned char content[PadSize];

  template <class T>
  T &getContent() {
    return *(reinterpret_cast<T *>(content));
  }
  template <class T>
  const T &getContent() const {
    return *(reinterpret_cast<const T *>(content));
  }

  using tid_Number = std::integral_constant<std::size_t, NodeType::Number>;
  using tid_String = std::integral_constant<std::size_t, NodeType::String>;
  using tid_Object = std::integral_constant<std::size_t, NodeType::Object>;
  using tid_Array = std::integral_constant<std::size_t, NodeType::Array>;
  using tid_Bool = std::integral_constant<std::size_t, NodeType::Bool>;
  using tid_Null = std::integral_constant<std::size_t, NodeType::Null>;

public:
  template <std::size_t typeIdx>
  jsonNode(std::integral_constant<std::size_t, typeIdx>) {
    nodeType = typeIdx;
  }

  ~jsonNode() {
    switch (nodeType) {
      case NodeType::String:
        getContent<jsString>().~jsString();
        break;
      case NodeType::Array:
        getContent<jsArray>().~jsArray();
        break;
      case NodeType::Object:
        getContent<jsObject>().~jsObject();
        break;
      default:
        break;
    }
  }

#if LIGHTJSON_RT_TYPE_CHECK
#define TYPE_CHECK(_Type, _TypeName)                                          \
  if (nodeType != NodeType::_Type)                                            \
    throw std::runtime_error("Value is not " _TypeName ". ");
#else
#define TYPE_CHECK(Type, TypeName) (void)(0)
#endif  // LIGHTJSON_RT_TYPE_CHECK

  /**
   * Return the type of this node.
   */
  std::size_t getType() const { return nodeType; }

  /**
   * Parse the current node as a number.
   * An exception is thrown if the current node is not a number.
   */
  void get(jsNumber &outNumber) const {
    TYPE_CHECK(Number, "a number");
    outNumber = getContent<jsNumber>();
  }

  /**
   * Pass the current as a number and downcast to an integer.
   */
  void get(int &outInt) const {
    TYPE_CHECK(Number, "a number");
    outInt = static_cast<int>(getContent<jsNumber>());
  }

  /**
   * Parse the current node as a string.
   * An exception is thrown if the current node is not a string.
   */
  void get(jsString &outStr) const {
    TYPE_CHECK(String, "a string");
    outStr = getContent<jsString>();
  }

  /**
   * Parse the current node as a Bool.
   * An exception is thrown if the current node is not a Bool.
   */
  void get(jsBool &outBool) const {
    TYPE_CHECK(Bool, "a boolean");
    outBool = getContent<jsBool>();
  }

  /**
   * A helper for parsing an array of values.
   * @tparam T Must be one of jsNumber, jsBool or jsString.
   */
  template <class T>
  void get(std::vector<T> &outVec) const {
    TYPE_CHECK(Array, "an array");
    const jsArray &theArray = getContent<jsArray>();
    T element;
    for (const auto &subNode : theArray) {
      subNode->get(element);
      outVec.push_back(element);
    }
  }

  /**
   * Parse the current node as a Vec<T_Num, *>.
   * A exception is thrown if Dim is less than the dimension of the array.
   */
#if LIGHTJSON_INTEROPERABILITY
  template <class T_Num, int Dim>
  void get(Vec<T_Num, Dim> &outVec) const {
    TYPE_CHECK(Array, "an array");
    const jsArray &theArray = getContent<jsArray>();
#if LIGHTJSON_RT_BOUND_CHECK
    if (theArray.size() > Dim)
      throw std::runtime_error("Mismatch dimension. ");
#endif  // LIGHTJSON_RT_BOUND_CHECK
    for (int d = 0; d < Dim; ++d)
      theArray[d]->get(outVec[d]);
  }
#endif  // LIGHTJSON_INTEROPERABILITY

  const jsArray &getArray() const {
    TYPE_CHECK(Array, "an array");
    return getContent<jsArray>();
  }

  /**
   * Parse the current node as an array and return its index-th element.
   * An exception is thrown if the current node is not an array.
   */
  const jsonNode &operator[](const std::size_t index) const {
    TYPE_CHECK(Array, "an array");
    const jsArray &theArr = getContent<jsArray>();
#if LIGHTJSON_RT_BOUND_CHECK
    if (index >= theArr.size())
      throw std::runtime_error("Array access out of bound. ");
#endif
    return *(theArr[index]);
  }

  const jsObject &getObject() const {
    TYPE_CHECK(Object, "an object");
    return getContent<jsObject>();
  }

  /**
   * Check whether the current object has a specific key.
   */
  bool has(const jsString &key) const {
    TYPE_CHECK(Object, "an object");
    const jsObject &theObj = getContent<jsObject>();
    auto it = std::lower_bound(theObj.cbegin(),
                               theObj.cend(),
                               key,
                               [](const jsMember &lhs, const jsString &rhs) {
                                 return lhs.first < rhs;
                               });
    return it != theObj.cend();
  }

  /**
   * Parse the current node as an object and inspect the value associated
   * with 'key'. An exception is thrown if the current node is not an object,
   * or no such 'key' exists.
   */
  const jsonNode &operator[](const jsString &key) const {
    TYPE_CHECK(Object, "an object");
    const jsObject &theObj = getContent<jsObject>();
    auto it = std::lower_bound(theObj.cbegin(),
                               theObj.cend(),
                               key,
                               [](const jsMember &lhs, const jsString &rhs) {
                                 return lhs.first < rhs;
                               });
#if LIGHTJSON_RT_BOUND_CHECK
    if (it == theObj.cend() || it->first != key)
      throw std::runtime_error(
          "The prescribed key is not found in the object. ");
#endif
    return *(it->second);
  }

  friend class jsonParser;
};

template <>
inline jsonNode::jsonNode(
    std::integral_constant<std::size_t, NodeType::String>) {
  nodeType = NodeType::String;
  new (content) jsString;
}

template <>
inline jsonNode::jsonNode(
    std::integral_constant<std::size_t, NodeType::Array>) {
  nodeType = NodeType::Array;
  new (content) jsArray;
}

template <>
inline jsonNode::jsonNode(
    std::integral_constant<std::size_t, NodeType::Object>) {
  nodeType = NodeType::Object;
  new (content) jsObject;
}

//======================================================================

/**
 * Manage the allocation of the small jsonNode objects.
 */
class jsonNodePool {
public:
  enum { NodeSize = sizeof(jsonNode) };
  enum { initBlockSize = 128 };
  using Byte = unsigned char;

  jsonNodePool() {
    curBlockSize = initBlockSize / 2;
    endOfBlock = nextAvail = nullptr;
  }

  template <class T>
  jsonNode *alloc(const T &t) {
    if (nextAvail == endOfBlock) {
      curBlockSize *= 2;
      Byte *newMem = new Byte[NodeSize * curBlockSize];
      headOfBlock.push_back(newMem);
      nextAvail = reinterpret_cast<jsonNode *>(newMem);
      endOfBlock = nextAvail + curBlockSize;
    }
    jsonNode *pNode = nextAvail++;
    new (pNode) jsonNode(t);
    return pNode;
  }

  ~jsonNodePool() {
    std::size_t blockSize = initBlockSize;
    for (auto &memHead : headOfBlock) {
      jsonNode *pNode = reinterpret_cast<jsonNode *>(memHead);
      for (std::size_t i = 0; i < blockSize; ++i, ++pNode)
        pNode->~jsonNode();
      delete[] memHead;
      blockSize *= 2;
    }
  }

  std::size_t getMemUsage() const { return (curBlockSize * 2 - 1) * NodeSize; }

  // disallow the copy & move operations
  jsonNodePool(const jsonNodePool &) = delete;
  jsonNodePool(jsonNodePool &&) = delete;
  jsonNodePool &operator=(const jsonNodePool &) = delete;
  jsonNodePool &operator=(jsonNodePool &&) = delete;

protected:
  std::vector<Byte *> headOfBlock;
  std::size_t curBlockSize;
  jsonNode *endOfBlock;
  jsonNode *nextAvail;
};

/**
 * The JSON parser.
 */
class jsonParser : protected jsonNodePool {
public:
  /**
   * The constructor.
   */
  jsonParser() { internalState = State::Value_Required; }

  /**
   * @pre The user should not call this function
   * before the parsing of the entire JSON script is complete.
   * @return The unique and the topmost value in a JSON script.
   */
  const jsonNode &getRoot() const {
    return *(reinterpret_cast<jsonNode *>(headOfBlock.front()));
  }

  /**
   * Parse a snippet of JSON script.
   * @pre Literals must not be interrupted.
   */
  void parse(const char *jsBuffer) {
    pos = jsBuffer;
    doParse();
  }

  /**
   * Signal the end of the JSON script.
   *
   * An exception is thrown if any syntax error is detected.
   */
  void finish() {
    if (internalState != State::Value_After || !pendingKey.empty() ||
        !nodeHier.empty()) {
      throw std::runtime_error("Incorrect internal state upon finishing. ");
    }
    //    std::cerr << "Mem usage = " << getMemUsage() << " bytes. " <<
    //    std::endl;
  }

protected:
  // =============================================================================

  /**
   * The states of the state machine.
   */
  enum class State {
    Object_Opened,
    Array_Opened,
    Keyword_Required,
    Keyword_After,
    Value_Required,
    Value_After
  };

  // The state machine
  const char *pos;
  State internalState;
  std::stack<jsString> pendingKey;
  std::stack<jsonNode *> nodeHier;

  /**
   * The main body of the parser.
   */
  void doParse() {
    while (1) {
      consumeWhitespace();
      bool isClosing = false;
      if (*pos == '\0')
        return;
      switch (internalState) {
        case State::Object_Opened:
          if (*pos == '}') {
            ++pos;
            isClosing = true;
          } else {
            internalState = State::Keyword_Required;
          }
          break;
        case State::Array_Opened:
          if (*pos == ']') {
            ++pos;
            isClosing = true;
          } else {
            internalState = State::Value_Required;
          }
          break;
        case State::Keyword_Required: {
          auto kw = parseString();
          pendingKey.push(kw);
          internalState = State::Keyword_After;
        } break;
        case State::Keyword_After:
          if (*pos != ':')
            throw std::runtime_error("Colon is required. ");
          ++pos;
          internalState = State::Value_Required;
          break;
        case State::Value_Required: {
          jsonNode *thisValue;
          // peek the next type
          if (*pos == '{') {
            // a new node and open object
            ++pos;
            thisValue = alloc(jsonNode::tid_Object());
            internalState = State::Object_Opened;
          } else if (*pos == '[') {
            // a new node and open array
            ++pos;
            thisValue = alloc(jsonNode::tid_Array());
            internalState = State::Array_Opened;
          } else {
            if (*pos == '-' || ::isdigit(*pos)) {
              thisValue = alloc(jsonNode::tid_Number());
              thisValue->getContent<jsNumber>() = parseNumber();
            } else if (*pos == '\"') {
              thisValue = alloc(jsonNode::tid_String());
              thisValue->getContent<jsString>() = parseString();
            } else {
              thisValue = alloc(jsonNode::tid_Bool());
              parseBuiltin(thisValue);
            }
            internalState = State::Value_After;
          }
          if (!nodeHier.empty()) {
            jsonNode *parentNode = nodeHier.top();
            if (parentNode->nodeType == NodeType::Array) {
              parentNode->getContent<jsArray>().push_back(thisValue);
            } else if (parentNode->nodeType == NodeType::Object) {
              assert(!pendingKey.empty());
              parentNode->getContent<jsObject>().push_back(
                  std::make_pair(pendingKey.top(), thisValue));
              pendingKey.pop();
            } else {
              assert(0);
            }
          }
          if (internalState == State::Object_Opened) {
            nodeHier.push(thisValue);
          } else if (internalState == State::Array_Opened) {
            nodeHier.push(thisValue);
          }
        } break;
        case State::Value_After:
          if (nodeHier.empty()) {
            throw std::runtime_error("Invalid content after the root value. ");
          } else {
            jsonNode *parent = nodeHier.top();
            std::size_t parentType = parent->nodeType;
            if (*pos == ',') {
              ++pos;
              internalState = (parentType == NodeType::Array)
                                  ? (State::Value_Required)
                                  : (State::Keyword_Required);
            } else if (*pos == ']') {
              if (parentType != NodeType::Array)
                throw std::runtime_error("Mismatched ]. ");
              ++pos;
              isClosing = true;
            } else if (*pos == '}') {
              if (parentType != NodeType::Object)
                throw std::runtime_error("Mismatched }. ");
              ++pos;
              isClosing = true;
              // sort the object by keys
              jsObject &theObj = parent->getContent<jsObject>();
              std::sort(theObj.begin(),
                        theObj.end(),
                        [](const jsMember &lhs, const jsMember &rhs) {
                          return lhs.first < rhs.first;
                        });
            } else {
              throw std::runtime_error("Invalid syntax after a value. ");
            }
          }
          break;
      }
      if (isClosing) {
        assert(!nodeHier.empty());
        nodeHier.pop();
        internalState = State::Value_After;
      }
    }
  }

protected:
  // =============================================================================
  // Some useful parsers
  jsNumber parseNumber() {
    char *nextPos;
    jsNumber res = strtod(pos, &nextPos);
    pos = nextPos;
    return res;
  }

  jsString parseString() {
    jsString resStr;
    char uhex;
    // 1. Skip the opening quote.
    //    if (*pos != '\"')
    //      throw std::runtime_error("Invalid string : miss opening quote.
    //      ");
    assert(*pos == '\"');
    ++pos;
    // 2. Translate each character.
    while (1) {
      // the end of the string
      if (*pos == '\0')
        throw std::runtime_error("Invalid string : miss closing quote. ");
      else if (*pos == '\"')
        break;
      // handle the escaped character
      else if (*pos == '\\') {
        ++pos;
        switch (*pos) {
          // sorted by the author's subjective precedence
          case 'n':
            resStr += '\n';
            break;
          case '\"':
            resStr += '\"';
            break;
          case '\\':
            resStr += '\\';
            break;
          case 't':
            resStr += '\t';
            break;
          case 'u':
            ++pos;
            for (int n = 0; n < 2; ++n) {
              uhex = 0;
              for (int i = 0; i < 2; ++i, ++pos) {
                if (*pos >= '0' && *pos <= '9')
                  uhex = (uhex << 4) | (*pos - '0');
                else if (*pos >= 'a' && *pos <= 'f')
                  uhex = (uhex << 4) | (0xa + *pos - 'a');
                else if (*pos >= 'A' && *pos <= 'F')
                  uhex = (uhex << 4) | (0xa + *pos - 'A');
                else
                  throw std::runtime_error("Invalid hex in a string. ");
              }
              resStr += uhex;
            }
            --pos;
            break;
          case '/':
            resStr += '/';
            break;
          case 'b':
            resStr += '\b';
            break;
          case 'f':
            resStr += '\f';
            break;
          case 'r':
            resStr += '\r';
            break;
          default:
            throw std::runtime_error("Invalid string : unsupported escaped "
                                     "character. ");
        }
        ++pos;
      } else {
        resStr += *pos;
        ++pos;
      }
    }
    // 3. Skip the closing quote.
    ++pos;
    return resStr;
  }

  void parseBuiltin(jsonNode *pNode) {
    if (strncmp(pos, "true", 4) == 0) {
      pos += 4;
      pNode->nodeType = NodeType::Bool;
      pNode->getContent<jsBool>() = true;
    } else if (strncmp(pos, "false", 5) == 0) {
      pos += 5;
      pNode->nodeType = NodeType::Bool;
      pNode->getContent<jsBool>() = false;
    } else if (strncmp(pos, "null", 4) == 0) {
      pos += 4;
      pNode->nodeType = NodeType::Null;
    } else {
      throw std::runtime_error("Unrecognized type. ");
    }
  }

  void consumeWhitespace() {
    while (::isspace(*pos))
      ++pos;
  }
};

}  // namespace lightJSON

#endif  // LIGHTJSON_H
