#pragma once

#include <string>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <iostream>

#include <c10/macros/Macros.h>

namespace c10
{
template <typename KeyType>
inline std::string KeyStrRepr(const KeyType& /**/) {
    return "[key type printing not support]";
}

template <>
inline std::string KeyStrRepr(const std::string& key) {
    return key;
}

enum RegistryPriority {
  REGISTRY_FALLBACK = 1,
  REGISTRY_DEFAULT = 2,
  REGISTRY_PREFERRED = 3,
};

/**
 * @brief A template class that allows one to register classes by keys.
 *
 * The keys are usually a std::string specifying the name, but can be anything
 * that can be used in a std::map.
 *
 * You should most likely not use the Registry class explicitly, but use the
 * helper macros below to declare specific registries as well as registering
 * objects.
 */
template <class SrcType, class ObjectPtrType, class... Args>
class Registry {
public:
    typedef std::function<ObjectPtrType(Args...)> Creator;

    Registry(bool warning = true)
        : registry_(), priority_(), terminate_(true), warning_(warning) {}

    void Register(
        const SrcType& key,
        Creator creator,
        const RegistryPriority priority = REGISTRY_DEFAULT) {

        std::lock_guard<std::mutex> lock(register_mutex_);
        if (registry_.count(key) != 0) {
            auto cur_priority = priority_[key];
            if (priority > cur_priority) {
                std::string warn_msg =
                    "Overwriting already registered item for key " + KeyStrRepr(key);
                std::cerr << warn_msg << std::endl;
                registry_[key] = creator;
                priority_[key] = priority;
            } else if (priority == cur_priority) {
                std::string err_msg =
                    "Key already registered with the same priority: " + KeyStrRepr(key);
                std::cerr << err_msg << std::endl;
                if (terminate_) {
                    std::exit(1);
                } else {
                    throw std::runtime_error(err_msg);
                }
            } else if (warning_) {
                std::string warn_msg =
                    "Higher priority item already registered, skipping registration of " +
                    KeyStrRepr(key);
                std::cerr << warn_msg << std::endl;
            }
        } else {
            registry_[key] = creator;
            priority_[key] = priority;
        }

    }

    void Register(const SrcType& key,
                  Creator creator,
                  const std::string& help_msg,
                  const RegistryPriority priority = REGISTRY_DEFAULT) {
        Register(key, creator, priority);
        help_message_[key] = help_msg;
    }

    inline bool Has(const SrcType& key) {
        return (registry_.count(key));
    }

    ObjectPtrType Create(const SrcType& key, Args... args) {
        if (registry_.count(key) == 0) {
            // Returns nullptr if the key is not registered.
            return nullptr;
        }
        return registry_[key](args...);
    }

    std::vector<SrcType> Keys() const {
        std::vector<SrcType> keys;
        for (const auto& it: registry_) {
            keys.push_back(it.first);
        }
        return keys;
    }

    inline const std::unordered_map<SrcType, std::string>& HelpMessage() const {
        return help_message_;
    }

    const char* HelpMessage(const SrcType& key) const {
        auto it = help_message_.find(key);
        if (it == help_message_.end()) {
            return nullptr;
        }
        return it->second.c_str();
    }

    void SetTerminate(bool terminate) {
        terminate_ = terminate;
    }
private:
    std::unordered_map<SrcType, Creator> registry_;
    std::unordered_map<SrcType, RegistryPriority> priority_;
    bool terminate_;
    const bool warning_;
    std::unordered_map<SrcType, std::string> help_message_;
    std::mutex register_mutex_;
    C10_DISABLE_COPY_AND_ASSIGN(Registry);
};

template <class SrcType, class ObjectPtrType, class... Args>
class Registerer {
public:
    explicit Registerer(
        const SrcType& key,
        const Registry<SrcType, ObjectPtrType, Args...>* registry,
        typename Registry<SrcType, ObjectPtrType, Args...>::Creator creator,
        const std::string& help_msg = "") {
        registry->Register(key, creator, help_msg);
    }

    explicit Registerer(
        const SrcType& key,
        const RegistryPriority priority,
        Registry<SrcType, ObjectPtrType, Args...>* registry,
        typename Registry<SrcType, ObjectPtrType, Args...>::Creator creator,
        const std::string& help_msg = "") {
        registry->Register(key, creator, help_msg, priority);
    }

    template <class DerivedType>
    static ObjectPtrType DefaultCreator(Args... args) {
        return ObjectPtrType(new DerivedType(args...));
    }
};

} // namespace c10

#define C10_DECLARE_TYPED_REGISTRY(RegistryName, SrcType, ObjectType, PtrType, ...) \
    ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*                   \
    RegistryName();                                                                 \
    typedef ::c10::Registerer<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>          \
      Registerer##RegistryName


#define C10_DEFINE_TYPED_REGISTRY(                                              \
    RegistryName, SrcType, ObjectType, PtrType, ...)                            \
    ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*               \
    RegistryName() {                                                            \
        static ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*    \
            registry = new ::c10::                                              \
                Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>();        \
        return registry;                                                        \
    }

#define C10_DEFINE_TYPED_REGISTRY_WITHOUT_WARNING(                            \
    RegistryName, SrcType, ObjectType, PtrType, ...)                          \
  C10_EXPORT ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*    \
  RegistryName() {                                                            \
    static ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>*      \
        registry =                                                            \
            new ::c10::Registry<SrcType, PtrType<ObjectType>, ##__VA_ARGS__>( \
                false);                                                       \
    return registry;                                                          \
  }

// Note(Yangqing): The __VA_ARGS__ below allows one to specify a templated
// creator with comma in its templated arguments.
#define C10_REGISTER_TYPED_CREATOR(RegistryName, key, ...)                  \
  static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key, RegistryName(), ##__VA_ARGS__);

#define C10_REGISTER_TYPED_CREATOR_WITH_PRIORITY(                           \
    RegistryName, key, priority, ...)                                       \
  static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key, priority, RegistryName(), ##__VA_ARGS__);

#define C10_REGISTER_TYPED_CLASS(RegistryName, key, ...)                    \
  static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key,                                                                  \
      RegistryName(),                                                       \
      Registerer##RegistryName::DefaultCreator<__VA_ARGS__>,                \
      ::c10::demangle_type<__VA_ARGS__>());

#define C10_REGISTER_TYPED_CLASS_WITH_PRIORITY(                             \
    RegistryName, key, priority, ...)                                       \
  static Registerer##RegistryName C10_ANONYMOUS_VARIABLE(g_##RegistryName)( \
      key,                                                                  \
      priority,                                                             \
      RegistryName(),                                                       \
      Registerer##RegistryName::DefaultCreator<__VA_ARGS__>,                \
      ::c10::demangle_type<__VA_ARGS__>());


// C10_DECLARE_REGISTRY and C10_DEFINE_REGISTRY are hard-wired to use
// std::string as the key type, because that is the most commonly used cases.
#define C10_DECLARE_REGISTRY(RegistryName, ObjectType, ...) \
  C10_DECLARE_TYPED_REGISTRY(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

#define C10_DEFINE_REGISTRY(RegistryName, ObjectType, ...) \
  C10_DEFINE_TYPED_REGISTRY(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

#define C10_DEFINE_REGISTRY_WITHOUT_WARNING(RegistryName, ObjectType, ...) \
  C10_DEFINE_TYPED_REGISTRY_WITHOUT_WARNING(                               \
      RegistryName, std::string, ObjectType, std::unique_ptr, ##__VA_ARGS__)

#define C10_DECLARE_SHARED_REGISTRY(RegistryName, ObjectType, ...) \
  C10_DECLARE_TYPED_REGISTRY(                                      \
      RegistryName, std::string, ObjectType, std::shared_ptr, ##__VA_ARGS__)

#define C10_DEFINE_SHARED_REGISTRY(RegistryName, ObjectType, ...) \
  C10_DEFINE_TYPED_REGISTRY(                                      \
      RegistryName, std::string, ObjectType, std::shared_ptr, ##__VA_ARGS__)

#define C10_DEFINE_SHARED_REGISTRY_WITHOUT_WARNING( \
    RegistryName, ObjectType, ...)                  \
  C10_DEFINE_TYPED_REGISTRY_WITHOUT_WARNING(        \
      RegistryName, std::string, ObjectType, std::shared_ptr, ##__VA_ARGS__)