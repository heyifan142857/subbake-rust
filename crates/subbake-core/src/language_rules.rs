use crate::entities::{TranslationMode, TranslationReadabilityDefaults};
use crate::languages::normalize_language_name;

#[cfg(test)]
const LANGUAGE_RULES_POLICY_VERSION: u64 = 1;

pub(crate) trait LanguageRule: Sync {
    fn id(&self) -> &'static str;
    fn version(&self) -> u64 {
        1
    }
    fn applies(&self, source: &str, target: &str) -> bool;
}

pub(crate) struct GenericRules;
pub(crate) struct EnglishRules;
pub(crate) struct JapaneseRules;
pub(crate) struct ChineseRules;
pub(crate) struct CjkRules;
pub(crate) struct EnglishToChineseRules;

impl LanguageRule for GenericRules {
    fn id(&self) -> &'static str {
        "generic"
    }

    fn applies(&self, _source: &str, _target: &str) -> bool {
        true
    }
}

impl LanguageRule for EnglishRules {
    fn id(&self) -> &'static str {
        "source-en"
    }

    fn applies(&self, source: &str, _target: &str) -> bool {
        source == "en" || source == "Auto"
    }
}

impl LanguageRule for JapaneseRules {
    fn id(&self) -> &'static str {
        "source-ja"
    }

    fn applies(&self, source: &str, _target: &str) -> bool {
        source == "ja" || source == "Auto"
    }
}

impl LanguageRule for ChineseRules {
    fn id(&self) -> &'static str {
        "target-zh"
    }

    fn applies(&self, _source: &str, target: &str) -> bool {
        primary_language(target) == "zh"
    }
}

impl LanguageRule for CjkRules {
    fn id(&self) -> &'static str {
        "script-cjk"
    }

    fn applies(&self, source: &str, target: &str) -> bool {
        source == "Auto" || is_cjk_language(source) || is_cjk_language(target)
    }
}

impl LanguageRule for EnglishToChineseRules {
    fn id(&self) -> &'static str {
        "pair-en-zh"
    }

    fn applies(&self, source: &str, target: &str) -> bool {
        source == "en" && primary_language(target) == "zh"
    }
}

static GENERIC_RULES: GenericRules = GenericRules;
static ENGLISH_RULES: EnglishRules = EnglishRules;
static JAPANESE_RULES: JapaneseRules = JapaneseRules;
static CHINESE_RULES: ChineseRules = ChineseRules;
static CJK_RULES: CjkRules = CjkRules;
static ENGLISH_TO_CHINESE_RULES: EnglishToChineseRules = EnglishToChineseRules;

static REGISTRY: [&dyn LanguageRule; 6] = [
    &GENERIC_RULES,
    &ENGLISH_RULES,
    &JAPANESE_RULES,
    &CHINESE_RULES,
    &CJK_RULES,
    &ENGLISH_TO_CHINESE_RULES,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedLanguageRules {
    source: String,
    target: String,
    active: Vec<(&'static str, u64)>,
}

pub(crate) struct LanguageRuleRegistry;

impl LanguageRuleRegistry {
    pub(crate) fn resolve(source: &str, target: &str) -> ResolvedLanguageRules {
        let source = normalize_language_name(source, true);
        let target = normalize_language_name(target, false);
        let active = REGISTRY
            .iter()
            .filter(|rule| rule.applies(&source, &target))
            .map(|rule| (rule.id(), rule.version()))
            .collect();
        ResolvedLanguageRules {
            source,
            target,
            active,
        }
    }
}

impl ResolvedLanguageRules {
    #[cfg(test)]
    pub(crate) fn source(&self) -> &str {
        &self.source
    }

    #[cfg(test)]
    pub(crate) fn target(&self) -> &str {
        &self.target
    }

    #[cfg(test)]
    pub(crate) fn has(&self, id: &str) -> bool {
        self.active.iter().any(|(active, _)| *active == id)
    }

    #[cfg(test)]
    pub(crate) fn semantic_key(&self) -> String {
        let rules = self
            .active
            .iter()
            .map(|(id, version)| format!("{id}@{version}"))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "language-rules-v{LANGUAGE_RULES_POLICY_VERSION}:{}>{}:{}",
            self.source, self.target, rules
        )
    }

    pub(crate) fn supports_english_coreference(&self) -> bool {
        self.source == "en"
    }

    pub(crate) fn target_requires_non_latin_name(&self) -> bool {
        matches!(
            primary_language(&self.target),
            "zh" | "ja" | "ko" | "ru" | "uk" | "ar" | "hi" | "th"
        )
    }

    pub(crate) fn target_requires_script_change(&self, source: &str) -> bool {
        match primary_language(&self.target) {
            "zh" => ChineseRules::requires_script_change(source),
            "ja" => JapaneseRules::requires_script_change(source),
            "ko" => source.chars().any(|character| {
                character.is_ascii_alphabetic()
                    || matches!(character, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{9fff}')
            }),
            "ru" | "uk" | "ar" | "hi" | "th" => source
                .chars()
                .any(|character| character.is_ascii_alphabetic()),
            _ => false,
        }
    }

    pub(crate) fn readability_defaults(
        &self,
        mode: TranslationMode,
    ) -> TranslationReadabilityDefaults {
        if mode != TranslationMode::Cinema {
            return TranslationReadabilityDefaults {
                max_characters_per_second: None,
                max_characters_per_line: None,
                max_lines: None,
            };
        }
        let cjk = is_cjk_language(&self.target);
        TranslationReadabilityDefaults {
            max_characters_per_second: Some(if cjk { 23.0 } else { 17.0 }),
            max_characters_per_line: Some(if cjk { 32 } else { 42 }),
            max_lines: Some(2),
        }
    }
}

impl EnglishRules {
    pub(crate) fn possessive_base(value: &str) -> Option<&str> {
        let value = value.trim();
        for suffix in ["'s", "'S", "’s", "’S"] {
            if let Some(base) = value.strip_suffix(suffix)
                && !base.is_empty()
            {
                return Some(base);
            }
        }
        for suffix in ['\'', '’'] {
            if let Some(base) = value.strip_suffix(suffix)
                && base.ends_with(['s', 'S'])
            {
                return Some(base);
            }
        }
        None
    }

    pub(crate) fn is_coreference_pronoun(word: &str) -> bool {
        matches!(
            word,
            "he" | "her" | "hers" | "him" | "his" | "she" | "their" | "them" | "they"
        )
    }

    pub(crate) fn digit(word: &str) -> Option<u128> {
        match word {
            "zero" | "oh" => Some(0),
            "one" => Some(1),
            "two" => Some(2),
            "three" => Some(3),
            "four" => Some(4),
            "five" => Some(5),
            "six" => Some(6),
            "seven" => Some(7),
            "eight" => Some(8),
            "nine" => Some(9),
            _ => None,
        }
    }

    pub(crate) fn small_number(word: &str) -> Option<u128> {
        match word {
            "ten" => Some(10),
            "eleven" => Some(11),
            "twelve" => Some(12),
            "thirteen" => Some(13),
            "fourteen" => Some(14),
            "fifteen" => Some(15),
            "sixteen" => Some(16),
            "seventeen" => Some(17),
            "eighteen" => Some(18),
            "nineteen" => Some(19),
            "twenty" => Some(20),
            "thirty" => Some(30),
            "forty" => Some(40),
            "fifty" => Some(50),
            "sixty" => Some(60),
            "seventy" => Some(70),
            "eighty" => Some(80),
            "ninety" => Some(90),
            _ => None,
        }
    }

    pub(crate) fn scale(word: &str) -> Option<u128> {
        match word {
            "hundred" => Some(100),
            "thousand" => Some(1_000),
            "million" => Some(1_000_000),
            "billion" => Some(1_000_000_000),
            "trillion" => Some(1_000_000_000_000),
            _ => None,
        }
    }
}

impl JapaneseRules {
    pub(crate) const HONORIFICS: [&str; 8] =
        ["ちゃん", "さん", "さま", "先生", "博士", "君", "様", "氏"];

    pub(crate) fn honorific_names(text: &str) -> Vec<String> {
        let mut names = Vec::new();
        for honorific in Self::HONORIFICS {
            for (honorific_at, _) in text.match_indices(honorific) {
                let mut start = honorific_at;
                let mut length = 0;
                for (index, character) in text[..honorific_at].char_indices().rev() {
                    if length == 8 || !Self::is_name_character(character) {
                        break;
                    }
                    start = index;
                    length += 1;
                }
                if length < 2 {
                    continue;
                }
                let candidate = &text[start..honorific_at];
                if !names.iter().any(|name| name == candidate) {
                    names.push(candidate.to_owned());
                }
            }
        }
        names
    }

    pub(crate) fn requires_script_change(source: &str) -> bool {
        source
            .chars()
            .any(|character| character.is_ascii_alphabetic())
    }

    fn is_name_character(character: char) -> bool {
        matches!(
            character,
            '\u{3400}'..='\u{9fff}'
                | '\u{30a0}'..='\u{30ff}'
                | '\u{ff66}'..='\u{ff9f}'
                | '·'
        )
    }
}

impl ChineseRules {
    pub(crate) fn requires_script_change(source: &str) -> bool {
        source.chars().any(|character| {
            character.is_ascii_alphabetic()
                || matches!(character, '\u{3040}'..='\u{30ff}' | '\u{ff66}'..='\u{ff9f}')
        })
    }
}

impl CjkRules {
    pub(crate) const DIGIT_SEQUENCE_CONTEXTS: &[&str] = &["年", "编号", "代码", "号码", "号"];

    pub(crate) const QUANTITY_CONTEXTS: &[&str] = &[
        "摄氏度",
        "华氏度",
        "个百分点",
        "个小时",
        "个星期",
        "个季度",
        "个世纪",
        "人民币",
        "分钟",
        "秒钟",
        "小时",
        "钟头",
        "星期",
        "季度",
        "世纪",
        "年代",
        "个月",
        "公里",
        "千米",
        "厘米",
        "毫米",
        "英里",
        "英尺",
        "英寸",
        "公斤",
        "千克",
        "毫升",
        "加仑",
        "美元",
        "美金",
        "欧元",
        "英镑",
        "日元",
        "韩元",
        "块钱",
        "等级",
        "编号",
        "代码",
        "号码",
        "个人",
        "点钟",
        "年",
        "岁",
        "月",
        "周",
        "天",
        "日",
        "时",
        "点",
        "刻",
        "分",
        "秒",
        "人",
        "个",
        "位",
        "名",
        "只",
        "条",
        "本",
        "件",
        "张",
        "辆",
        "艘",
        "架",
        "台",
        "部",
        "枚",
        "颗",
        "块",
        "份",
        "家",
        "间",
        "所",
        "场",
        "次",
        "遍",
        "回",
        "趟",
        "轮",
        "集",
        "季",
        "章",
        "页",
        "行",
        "句",
        "字",
        "层",
        "级",
        "号",
        "届",
        "期",
        "队",
        "组",
        "对",
        "双",
        "套",
        "种",
        "米",
        "码",
        "吨",
        "磅",
        "斤",
        "升",
        "克",
        "元",
        "档",
        "阶",
        "星",
        "倍",
        "度",
    ];

    pub(crate) fn is_character(character: char) -> bool {
        matches!(
            character as u32,
            0x1100..=0x11ff
                | 0x2e80..=0x2fdf
                | 0x3040..=0x30ff
                | 0x31f0..=0x31ff
                | 0x3400..=0x4dbf
                | 0x4e00..=0x9fff
                | 0xac00..=0xd7af
                | 0xf900..=0xfaff
                | 0xff66..=0xff9d
        )
    }

    pub(crate) fn is_han_character(character: char) -> bool {
        matches!(
            character,
            '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{f900}'..='\u{faff}'
        )
    }

    pub(crate) fn is_numeral(character: char) -> bool {
        Self::digit(character).is_some() || matches!(character, '十' | '百' | '千' | '万' | '亿')
    }

    pub(crate) fn digit(character: char) -> Option<u128> {
        match character {
            '零' | '〇' => Some(0),
            '一' => Some(1),
            '二' | '两' => Some(2),
            '三' => Some(3),
            '四' => Some(4),
            '五' => Some(5),
            '六' => Some(6),
            '七' => Some(7),
            '八' => Some(8),
            '九' => Some(9),
            _ => None,
        }
    }

    pub(crate) fn unit_value(character: char) -> Option<u128> {
        match character {
            '十' => Some(10),
            '百' => Some(100),
            '千' => Some(1_000),
            '万' => Some(10_000),
            '亿' => Some(100_000_000),
            _ => None,
        }
    }
}

fn primary_language(value: &str) -> &str {
    value.split('-').next().unwrap_or_default()
}

fn is_cjk_language(value: &str) -> bool {
    matches!(primary_language(value), "zh" | "ja" | "ko")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_registered_rules_in_stable_order() {
        let english_chinese = LanguageRuleRegistry::resolve("English", "zh-Hans");
        assert_eq!(english_chinese.source(), "en");
        assert_eq!(english_chinese.target(), "zh-Hans");
        assert!(english_chinese.has("source-en"));
        assert!(english_chinese.has("target-zh"));
        assert!(english_chinese.has("script-cjk"));
        assert!(english_chinese.has("pair-en-zh"));
        assert_eq!(
            english_chinese.semantic_key(),
            "language-rules-v1:en>zh-Hans:generic@1,source-en@1,target-zh@1,script-cjk@1,pair-en-zh@1"
        );

        let auto_chinese = LanguageRuleRegistry::resolve("Auto", "Chinese");
        assert!(auto_chinese.has("source-en"));
        assert!(auto_chinese.has("source-ja"));
        assert!(!auto_chinese.has("pair-en-zh"));
    }

    #[test]
    fn preserves_target_script_and_readability_behavior() {
        let chinese = LanguageRuleRegistry::resolve("en", "zh-Hans");
        assert!(chinese.target_requires_script_change("Alice"));
        assert!(chinese.target_requires_script_change("アリス"));
        assert!(!chinese.target_requires_script_change("爱丽丝"));
        assert_eq!(
            chinese
                .readability_defaults(TranslationMode::Cinema)
                .max_characters_per_line,
            Some(32)
        );

        let french = LanguageRuleRegistry::resolve("en", "fr");
        assert!(!french.target_requires_script_change("Alice"));
        assert_eq!(
            french
                .readability_defaults(TranslationMode::Cinema)
                .max_characters_per_line,
            Some(42)
        );
    }
}
